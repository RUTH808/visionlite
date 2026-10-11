"""VisionLite: compact CNNs, distillation and measured CPU inference."""
import argparse,copy,gzip,hashlib,json,time,warnings
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from reporting import download,save_json,report,table,environment
CLASSES=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
CHECKSUMS={'train-images-idx3-ubyte.gz':'8d4fb7e6c68d591d4c3dfef9ec88bf0d','train-labels-idx1-ubyte.gz':'25c81989df183df01b3e8a0aad5dffbe','t10k-images-idx3-ubyte.gz':'bef4ecab320f06d8554ea6380940ec79','t10k-labels-idx1-ubyte.gz':'bb300cfdad3c16e7a12a480ee83cd310'}

class CNN(nn.Module):
    def __init__(self,width=16,hidden=64):
        super().__init__();self.features=nn.Sequential(nn.Conv2d(1,width,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(width,width*2,3,padding=1),nn.ReLU(),nn.MaxPool2d(2));self.head=nn.Sequential(nn.Flatten(),nn.Linear(width*2*7*7,hidden),nn.ReLU(),nn.Linear(hidden,10))
    def forward(self,x):return self.head(self.features(x))

def load_data():
    arrays={}
    for file,md5 in CHECKSUMS.items():
        path=download('https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'+file,'data/'+file)
        if hashlib.md5(path.read_bytes()).hexdigest()!=md5:raise ValueError('Dataset checksum mismatch: '+file)
        raw=gzip.decompress(path.read_bytes());offset=16 if 'images' in file else 8;a=np.frombuffer(raw,dtype=np.uint8).copy()
        arrays[file]=a[offset:].reshape(-1,1,28,28) if 'images' in file else a[offset:]
    return arrays['train-images-idx3-ubyte.gz'],arrays['train-labels-idx1-ubyte.gz'],arrays['t10k-images-idx3-ubyte.gz'],arrays['t10k-labels-idx1-ubyte.gz']

@torch.inference_mode()
def probabilities(model,x,batch=256):
    model.eval();return torch.cat([model(x[i:i+batch]).softmax(1) for i in range(0,len(x),batch)]).numpy()

def fit(model,x,y,xval,yval,epochs,teacher=None):
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    g=torch.Generator().manual_seed(42);loader=DataLoader(TensorDataset(x,y),batch_size=256,shuffle=True,generator=g)
    best=-1;state=None;history=[];start=time.perf_counter()
    if teacher is not None:teacher.eval()
    for epoch in range(epochs):
        model.train();total=0.
        for batch,target in loader:
            optimizer.zero_grad();logits=model(batch);ce=nn.functional.cross_entropy(logits,target)
            if teacher is not None:
                with torch.no_grad():soft=teacher(batch)
                kd=nn.functional.kl_div(nn.functional.log_softmax(logits/3,dim=1),nn.functional.softmax(soft/3,dim=1),reduction='batchmean')*9
                loss=.5*ce+.5*kd
            else:loss=ce
            loss.backward();optimizer.step();total+=float(loss.detach())*len(batch)
        acc=float(accuracy_score(yval,probabilities(model,xval).argmax(1)));history.append({'epoch':epoch+1,'training_loss':total/len(x),'validation_accuracy':acc})
        print('epoch',epoch+1,'validation',round(acc,4),flush=True)
        if acc>best:best=acc;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);model.eval();return history,time.perf_counter()-start

def evaluate(epochs,output,train_limit=None):
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);Path('artifacts').mkdir(exist_ok=True)
    xa,ya,xt,yt=load_data();tr,va=train_test_split(np.arange(len(ya)),test_size=5000,random_state=42,stratify=ya)
    if train_limit:tr,_=train_test_split(tr,train_size=train_limit,stratify=ya[tr],random_state=42)
    X=torch.tensor(xa[tr]/255,dtype=torch.float32);Y=torch.tensor(ya[tr],dtype=torch.long);V=torch.tensor(xa[va]/255,dtype=torch.float32);T=torch.tensor(xt/255,dtype=torch.float32)
    teacher=CNN(32,128);th,tt=fit(teacher,X,Y,V,ya[va],epochs)
    torch.manual_seed(99);student=CNN(8,64);initial=copy.deepcopy(student.state_dict());sh,st=fit(student,X,Y,V,ya[va],epochs)
    distilled=CNN(8,64);distilled.load_state_dict(initial);dh,dt=fit(distilled,X,Y,V,ya[va],epochs,teacher=teacher)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');quantized=torch.ao.quantization.quantize_dynamic(copy.deepcopy(distilled),{nn.Linear},dtype=torch.qint8)
    models={'teacher':teacher,'student_supervised':student,'student_distilled':distilled,'student_int8_linear':quantized}
    torch.set_num_threads(1);rows=[];details={};noise=np.random.default_rng(2026).normal(0,.10,T.shape).astype('float32');corrupted=torch.clamp(T+torch.from_numpy(noise),0,1)
    for name,m in models.items():
        prob=probabilities(m,T);pred=prob.argmax(1);noisy=probabilities(m,corrupted).argmax(1);torch.save(m.state_dict(),'artifacts/'+name+'.pt');nbytes=Path('artifacts/'+name+'.pt').stat().st_size
        for _ in range(20):probabilities(m,T[:1])
        durations=[]
        for _ in range(200):
            start=time.perf_counter();probabilities(m,T[:1]);durations.append((time.perf_counter()-start)*1000)
        throughput=[]
        for _ in range(20):
            start=time.perf_counter();probabilities(m,T[:256]);throughput.append(256/(time.perf_counter()-start))
        acc=float(accuracy_score(yt,pred));n=len(yt);z=1.96;den=1+z*z/n;center=(acc+z*z/(2*n))/den;half=z*np.sqrt(acc*(1-acc)/n+z*z/(4*n*n))/den
        rows.append({'model':name,'test_accuracy':acc,'macro_f1':float(f1_score(yt,pred,average='macro')),'accuracy_ci_low':center-half,'accuracy_ci_high':center+half,'noise_accuracy':float(accuracy_score(yt,noisy)),'checkpoint_bytes':nbytes,'p50_ms':float(np.median(durations)),'p95_ms':float(np.quantile(durations,.95)),'batch256_images_per_second':float(np.median(throughput))})
        details[name]={'confusion_matrix':confusion_matrix(yt,pred).tolist(),'per_class_accuracy':{CLASSES[i]:float((pred[yt==i]==i).mean()) for i in range(10)}}
    m={'dataset':'Fashion-MNIST','data_checksums_md5':CHECKSUMS,'seed':42,'student_initialization_seed':99,'epochs_per_model':epochs,'splits':{'train':len(tr),'validation':len(va),'test':len(yt)},'teacher_parameters':sum(p.numel() for p in teacher.parameters()),'student_parameters':sum(p.numel() for p in distilled.parameters()),'distillation':{'temperature':3,'supervised_weight':.5},'training_seconds':{'teacher':tt,'student_supervised':st,'student_distilled':dt},'validation_histories':{'teacher':th,'student_supervised':sh,'student_distilled':dh},'test_results':rows,'details':details,'latency_protocol':'CPU 1 thread; batch 1; 20 warmups; 200 repetitions including softmax; P50/P95; throughput batch=256 over 20 repetitions; excludes loading and network','environment':environment(),'limitations':'One training seed; test accuracy intervals are Wilson binomial 95% intervals, not seed variability. Fashion-MNIST is 28x28 grayscale; performance does not establish real catalog-image performance. Quantization affects Linear layers only; convolutions remain FP32.'}
    save_json(output/'metrics.json',m);save_json(output/'split_indices.json',{'train':tr.tolist(),'validation':va.tolist()});save_json(output/'example_request.json',{'pixels':xt[0,0].tolist()})
    report(output/'dashboard.html','VisionLite','Measure the accuracy, size and CPU latency of compact image classifiers.',[('Held-out classification',table(rows,['model','test_accuracy','macro_f1','accuracy_ci_low','accuracy_ci_high','noise_accuracy'])),('Size and inference cost',table(rows,['model','checkpoint_bytes','p50_ms','p95_ms','batch256_images_per_second'])+'<p>'+m['latency_protocol']+'</p>'),('Training and validation',table([{'model':name,**h} for name,hist in m['validation_histories'].items() for h in hist],['model','epoch','training_loss','validation_accuracy'])),('Scope and limitations','<p>'+m['limitations']+'</p><p>Noise evaluation adds Gaussian noise with sigma 0.10 to normalized test images and clips to [0,1]. It is a single controlled stress test. Student models share initialization and data order. Checkpoints are selected on validation accuracy; test data are scored only after training.</p>')],{'Test images':f'{len(yt):,}','Classes':10,'Student parameters':f"{m['student_parameters']:,}"})
    print(json.dumps(m,indent=2));return m

def load_model(variant='student_int8_linear'):
    model=CNN(8,64)
    if variant=='student_int8_linear':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');model=torch.ao.quantization.quantize_dynamic(model,{nn.Linear},dtype=torch.qint8)
    elif variant!='student_distilled':raise ValueError('Unknown student variant')
    model.load_state_dict(torch.load('artifacts/'+variant+'.pt',weights_only=True));model.eval();return model

def predict(payload,model=None):
    if not isinstance(payload,dict) or 'pixels' not in payload:raise ValueError('Expected pixels: 28x28 grayscale values in [0,255]')
    a=np.asarray(payload['pixels'],dtype=np.float32)
    if a.shape!=(28,28) or not np.isfinite(a).all() or a.min()<0 or a.max()>255:raise ValueError('pixels must be a 28x28 finite array in [0,255]')
    if model is None:
        model=load_model()
    prob=probabilities(model,torch.tensor(a.reshape(1,1,28,28)/255))[0];i=int(prob.argmax());return {'class':CLASSES[i],'confidence':float(prob[i]),'probabilities':dict(zip(CLASSES,map(float,prob)))}

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','predict','serve']);p.add_argument('--epochs',type=int,default=5);p.add_argument('--train-limit',type=int);p.add_argument('--output',default='results');p.add_argument('--input');p.add_argument('--port',type=int,default=8013);a=p.parse_args()
    if a.command=='train':evaluate(a.epochs,a.output,a.train_limit)
    elif a.command=='predict':print(json.dumps(predict(json.loads(Path(a.input).read_text())),indent=2))
    else:
        from service import serve
        torch.set_num_threads(1);model=load_model();serve(lambda p:predict(p,model),a.port,'VisionLite')
if __name__=='__main__':main()

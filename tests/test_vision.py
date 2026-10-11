import numpy as np,torch,pytest
from app import CNN,predict,probabilities

def test_logits_and_probabilities():
 m=CNN(8,64);x=torch.rand(3,1,28,28);assert m(x).shape==(3,10)
 p=probabilities(m,x);assert np.allclose(p.sum(1),1,atol=1e-6)

def test_input_validation():
 for a in [[[0]],[[300]*28]*28,[[float('nan')]*28]*28]:
  with pytest.raises(ValueError):predict({'pixels':a})

def test_compact_model_is_smaller():
 assert sum(p.numel() for p in CNN(8,64).parameters())<sum(p.numel() for p in CNN(32,128).parameters())/5

def test_inference_label_and_confidence():
 r=predict({'pixels':np.zeros((28,28)).tolist()},CNN(8,64));assert 0<=r['confidence']<=1 and len(r['probabilities'])==10

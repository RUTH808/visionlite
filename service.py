"""Local JSON inference server. Bind localhost; no auth or production claims."""
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

def serve(predict,port,title):
 class Handler(BaseHTTPRequestHandler):
  def send(self,status,payload):
   data=json.dumps(payload,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
  def do_GET(self):
   if self.path=='/health':self.send(200,{'status':'ok','service':title})
   elif self.path in ['/', '/report', '/example']:
    from pathlib import Path
    targets={'/':('web/index.html','text/html; charset=utf-8'),'/report':('results/dashboard.html','text/html; charset=utf-8'),'/example':('results/example_request.json','application/json')}
    path,mime=targets[self.path]
    if not Path(path).exists():self.send(404,{'error':'Run training/evaluation first'});return
    data=Path(path).read_bytes();self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
   else:self.send(404,{'error':'Use POST /predict; GET /health'})
  def do_POST(self):
   if self.path!='/predict':self.send(404,{'error':'Not found'});return
   try:
    n=int(self.headers.get('Content-Length','0'))
    if not 0<n<=65536:raise ValueError('JSON body must be 1-65536 bytes')
    payload=json.loads(self.rfile.read(n));result=predict(payload);self.send(200,result)
   except (ValueError,TypeError,KeyError) as e:self.send(400,{'error':str(e)})
  def setup(self):
   super().setup();self.connection.settimeout(15)
 print(f'{title} listening on http://127.0.0.1:{port}',flush=True)
 ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()

import json,os,sys,urllib.request
BASE=os.environ["TENSOR_API_BASE"];TOK=open(".token").read().strip()
p=sys.argv[1]; lim=int(sys.argv[2]) if len(sys.argv)>2 else 3500
req=urllib.request.Request(BASE+"/"+p,headers={"Authorization":"Bearer "+TOK})
try:
    d=json.loads(urllib.request.urlopen(req,timeout=30).read().decode())
except Exception as e:
    print("ERR",e); sys.exit()
print(json.dumps(d,ensure_ascii=False)[:lim])

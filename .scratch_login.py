import json,os,urllib.request
BASE=os.environ.get("TENSOR_API_BASE","http://127.0.0.1:8010/api/v1")
u=os.environ["TENSOR_AGENT_USER"]; p=os.environ["TENSOR_AGENT_PASS"]
def post(path,body,hdrs=None):
    req=urllib.request.Request(BASE+"/"+path,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json",**(hdrs or {})},method="POST")
    try:
        return json.loads(urllib.request.urlopen(req,timeout=20).read().decode())
    except Exception as e:
        return {"err":str(e)}
for path in ("auth/login","tensor/auth/login","login"):
    r=post(path,{"username":u,"password":p})
    if "err" not in r:
        tok=r.get("access_token") or r.get("token")
        print(path,"OK token_len",len(tok) if tok else None, list(r)[:6])
        open(".token","w").write(tok or "")
        break
    else:
        print(path,r["err"])

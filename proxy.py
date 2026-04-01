from flask import Flask,request,Response
import requests,re
app=Flask(__name__)
HF="https://okezue-pimorph-labeler.hf.space"
@app.route("/",defaults={"path":""})
@app.route("/<path:path>",methods=["GET","POST","PUT","DELETE","PATCH"])
def proxy(path):
    url=f"{HF}/{path}"
    h={k:v for k,v in request.headers if k.lower() not in ("host","cookie")}
    r=requests.request(request.method,url,headers=h,
                       data=request.get_data(),cookies=request.cookies,params=request.args,
                       allow_redirects=False,stream=True,timeout=600)
    headers=[]
    for k,v in r.raw.headers.items():
        kl=k.lower()
        if kl in ("transfer-encoding","connection"): continue
        if kl=="set-cookie":
            v=re.sub(r';\s*[Dd]omain=[^;]*','',v)
            v=re.sub(r';\s*[Ss]ecure','',v)
            v=re.sub(r';\s*[Ss]ame[Ss]ite=[^;]*','',v)
        if kl=="location":
            v=v.replace(HF,"")
        headers.append((k,v))
    return Response(r.iter_content(8192),status=r.status_code,headers=headers,content_type=r.headers.get("content-type"))

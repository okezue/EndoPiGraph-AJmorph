from flask import Flask,request,Response
import requests
app=Flask(__name__)
HF="https://okezue-pimorph-labeler.hf.space"
@app.route("/",defaults={"path":""})
@app.route("/<path:path>",methods=["GET","POST","PUT","DELETE","PATCH"])
def proxy(path):
    url=f"{HF}/{path}"
    r=requests.request(request.method,url,headers={k:v for k,v in request.headers if k.lower()!="host"},
                       data=request.get_data(),cookies=request.cookies,params=request.args,
                       allow_redirects=False,stream=True,timeout=600)
    headers=[(k,v) for k,v in r.raw.headers.items() if k.lower() not in ("transfer-encoding","connection")]
    return Response(r.iter_content(8192),status=r.status_code,headers=headers,content_type=r.headers.get("content-type"))

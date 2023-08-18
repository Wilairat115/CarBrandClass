import json
import os
import pickle

import requests
import xgboost
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# use docker
from app.code import predict_brand

# from code import predict_brand

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)
# model = pickle.load(open(r'C:\Users\User\Desktop\AI\Assignment1\CarBrandClass\model\model_XGB.pkl','rb'))
# use docker
model = pickle.load(open(os.getcwd()+f'/model/model_XGB.pk','rb'))

# end_hog = 'http://localhost:8080/api/gethog'
# use docker
end_hog = 'http://172.17.0.2:80/api/gethog'

@app.get("/")
def root():
    return {"message": "This is my api"}

@app.post("/api/carbrand")
async def read_str(request:Request):
    item = await request.json()
    datas = item["img"]
    jssend = {"item_str":datas}
    
    hog = requests.post(url=end_hog,json=jssend)
    # res = predict_brand(model,hog)
    # return {"result":res }
    res = predict_brand(model,hog.json()['HOG'])
    return res
    
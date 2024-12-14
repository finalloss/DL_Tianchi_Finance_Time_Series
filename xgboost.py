import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv("../data/predict_table.csv") 
data2 = pd.read_csv("../data/product_info_simple_final_train.csv") 
from datetime import datetime
base = int(datetime.strptime(str(20210104), "%Y%m%d").timestamp())+58320000
change = lambda x:(int(datetime.strptime(str(x), "%Y%m%d").timestamp())-base)/86400
models = {}
models2 = {}
models3 = {}

import xgboost
for i in data['product_pid'].unique():
    X=data2[data2['product_pid']==i]["transaction_date"].apply(lambda x:change(x)).values.reshape(-1,1)
    y=data2[data2['product_pid']==i]["apply_amt"]
    y2=data2[data2['product_pid']==i]["redeem_amt"]
    y3=data2[data2['product_pid']==i]["net_in_amt"]
    models[i]=xgboost.XGBRegressor()
    models[i].fit(X,y)
    models2[i]=xgboost.XGBRegressor()
    models2[i].fit(X,y2)
    models3[i]=xgboost.XGBRegressor()
    models3[i].fit(X,y3)

for i in data["product_pid"].unique():
    X=data[data['product_pid']==i]["transaction_date"].apply(lambda x:change(x)).values.reshape(-1,1)
    pre = models[i].predict(X)
    pre2 = models2[i].predict(X)
    pre3 = models3[i].predict(X)
    data.loc[data['product_pid']==i,"apply_amt_pred"] = pre
    data.loc[data['product_pid']==i,"redeem_amt_pred"] = pre2
    data.loc[data['product_pid']==i,"net_in_amt_pred"] = pre3
data["net_in_amt_pred"] = (data["apply_amt_pred"]-data["redeem_amt_pred"])*0.5+data["net_in_amt_pred"]*0.5
data.to_csv("predict_table.csv",index=False)
data
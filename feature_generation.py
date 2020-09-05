import pandas as pd
from dataExploration import distReports,plotGrabh
import numpy as np
from iv import iv_all,binning
import matplotlib.pyplot as plt
from funcs import crossVariablelowRam
import time

import seaborn as sns
start = time.time()
for i in range(10):
    if i ==0 :
        main=pd.read_csv('/home/pooja/PycharmProjects/datanalysis/featureEngeering/train'+str(i)+".csv")
        main.to_csv('/home/pooja/PycharmProjects/datanalysis/featureEngeering/train'+".csv")
    else :
        main = pd.read_csv('/home/pooja/PycharmProjects/datanalysis/featureEngeering/train' + str(i) + ".csv")
        main.to_csv('/home/pooja/PycharmProjects/datanalysis/featureEngeering/train'  + ".csv",mode = 'a', header = False)
if __name__ == "__main__":
    job=2
    #s_POS_CASH_balance=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets//POS_CASH_balance.csv")
    data = pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets/train.csv")
    target = data.set_index('SK_ID_CURR')[['TARGET']]
    if job==1:


        x=crossVariablelowRam(data.drop('TARGET',axis=1).set_index('SK_ID_CURR'),target,ignoreList=['SK_ID_PREV','Unnamed: 0','SK_ID_CURR','TARGET'],target='TARGET',batch=10,loc="/home/pooja/PycharmProjects/datanalysis/featureEngeering/train")
    if job == 2:
        prev = pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets/treated/previous_application.csv", nrows=10000)
        target = data.set_index('SK_ID_CURR')[['TARGET']]
        x = crossVariablelowRam(prev.set_index('SK_ID_CURR'), target,
                                ignoreList=['SK_ID_PREV', 'Unnamed: 0', 'SK_ID_CURR', 'TARGET'], target='TARGET',
                                batch=10, loc="/home/pooja/PycharmProjects/datanalysis/featureEngeering/prev",groupByKey='SK_ID_CURR')
    if job ==3:
        data = pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets/train.csv", nrows=10)
        target = data.set_index('SK_ID_CURR')[['TARGET']]
        x = crossVariablelowRam(data.drop('TARGET', axis=1).set_index('SK_ID_CURR'), target,
                                ignoreList=['SK_ID_PREV', 'Unnamed: 0', 'SK_ID_CURR', 'TARGET'], target='TARGET',
                                batch=10, loc="/home/pooja/PycharmProjects/datanalysis/featureEngeering/train")
    #data=data[['EXT_SOURCE_3','EXT_SOURCE_2','EXT_SOURCE_1','TARGET','SK_ID_CURR']]
    #data['check']=1
    #s_POS_CASH_balance=s_POS_CASH_balance.join(data[['SK_ID_CURR','TARGET']].st_index('SK_ID_CURR'),on='SK_ID_CURR')
    #x=crossVariablelowRam(data.drop('TARGET',axis=1).set_index('SK_ID_CURR'),target,ignoreList=['SK_ID_PREV','Unnamed: 0','SK_ID_CURR','TARGET'],target='TARGET',batch=10,loc="/home/pooja/PycharmProjects/datanalysis/featureEngeering/train")
    #q=past(s_POS_CASH_balance,'SK_ID_CURR','MONTHS_BALANCE')
    pass
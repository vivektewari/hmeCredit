import pandas as pd
from dataExploration import distReports,plotGrabh
import numpy as np
from iv import iv_all,binning
import matplotlib.pyplot as plt
from funcs import crossVariablelowRam,normalize
import time

import seaborn as sns
start = time.time()
# for i in range(10):
#     if i ==0 :
#         main=pd.read_csv('/home/pooja/PycharmProjects/datanalysis/featureEngeering/train'+str(i)+".csv")
#         main.to_csv('/home/pooja/PycharmProjects/datanalysis/featureEngeering/train'+".csv")
#     else :
#         main = pd.read_csv('/home/pooja/PycharmProjects/datanalysis/featureEngeering/train' + str(i) + ".csv")
#         main.to_csv('/home/pooja/PycharmProjects/datanalysis/featureEngeering/train'  + ".csv",mode = 'a', header = False)
if __name__ == "__main__":
    job=3
    #s_POS_CASH_balance=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets//POS_CASH_balance.csv")
    data = pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets/train.csv")
    target = data.set_index('SK_ID_CURR')[['TARGET']]
    if job==1:


        x=crossVariablelowRam(data.drop('TARGET',axis=1).set_index('SK_ID_CURR'),target,ignoreList=['SK_ID_PREV','Unnamed: 0','SK_ID_CURR','TARGET'],target='TARGET',batch=10,loc="/home/pooja/PycharmProjects/datanalysis/featureEngeering/train")
    if job == 2:
        prev = pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets/previous_application.csv")
        x = crossVariablelowRam(prev.set_index('SK_ID_CURR'), target,
                                ignoreList=['SK_ID_PREV', 'Unnamed: 0', 'SK_ID_CURR', 'TARGET'], target='TARGET',
                                batch=2, loc="/home/pooja/PycharmProjects/datanalysis/featureEngeering/prev",groupByKey='SK_ID_CURR')
    if job ==3:
        data = pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets/POS_CASH_balance.csv")

        x = crossVariablelowRam(data.set_index('SK_ID_CURR'), target,
                                ignoreList=['SK_ID_PREV', 'Unnamed: 0', 'SK_ID_CURR', 'TARGET'], target='TARGET',
                                batch=2, loc="/home/pooja/PycharmProjects/datanalysis/featureEngeering/pos",groupByKey='SK_ID_CURR')
    if job ==3:
        data = pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets/credit_card_balance.csv")
        x = crossVariablelowRam(data.set_index('SK_ID_CURR'), target,
                                ignoreList=['SK_ID_PREV', 'Unnamed: 0', 'SK_ID_CURR', 'TARGET'], target='TARGET',
                                batch=2, loc="/home/pooja/PycharmProjects/datanalysis/featureEngeering/credit",groupByKey='SK_ID_CURR')
    if job == 3:
        data = pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets/bureau.csv")
        x = crossVariablelowRam(data.set_index('SK_ID_CURR'), target,
                                ignoreList=['SK_ID_BUREAU', 'Unnamed: 0', 'SK_ID_CURR', 'TARGET'], target='TARGET',
                                batch=4, loc="/home/pooja/PycharmProjects/datanalysis/featureEngeering/bureau",groupByKey='SK_ID_CURR')

    if job == 3:
        data1 = pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets/bureau.csv")
        data = pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets/bureau_balance.csv")
        data=data.set_index('SK_ID_BUREAU').join(data1.set_index('SK_ID_BUREAU')[['SK_ID_CURR']])
        data['bucket'] = data['STATUS'].apply(lambda row: int(row) if row not in ['C', 'X'] else 0)
        x = crossVariablelowRam(data.set_index('SK_ID_CURR'), target,
                                ignoreList=['SK_ID_BUREAU', 'Unnamed: 0', 'SK_ID_CURR', 'TARGET'], target='TARGET',
                                batch=2, loc="/home/pooja/PycharmProjects/datanalysis/featureEngeering/bureaubal",groupByKey='SK_ID_CURR')


    #data=data[['EXT_SOURCE_3','EXT_SOURCE_2','EXT_SOURCE_1','TARGET','SK_ID_CURR']]
    #data['check']=1
    #s_POS_CASH_balance=s_POS_CASH_balance.join(data[['SK_ID_CURR','TARGET']].st_index('SK_ID_CURR'),on='SK_ID_CURR')
    #x=crossVariablelowRam(data.drop('TARGET',axis=1).set_index('SK_ID_CURR'),target,ignoreList=['SK_ID_PREV','Unnamed: 0','SK_ID_CURR','TARGET'],target='TARGET',batch=10,loc="/home/pooja/PycharmProjects/datanalysis/featureEngeering/train")
    #q=past(s_POS_CASH_balance,'SK_ID_CURR','MONTHS_BALANCE')
    pass
    if job==20:
        prev = pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets/previous_application.csv")
        prev=prev.set_index('SK_ID_CURR')[['AMT_ANNUITY','AMT_APPLICATION']]
        #prev['mult']=prev['AMT_ANNUITY']*prev['AMT_APPLICATION']
        prev['div'] =  prev['AMT_APPLICATION']/prev['AMT_ANNUITY']
        #prev=prev.round(2)
        b = prev.dtypes
        prev=prev.replace(np.inf,np.nan)
        #prev=normalize(prev)
        #prev    =prev.astype(np.float32)
        prev=prev.join(target)
        prev=prev.groupby('SK_ID_CURR').agg(['min', 'max','sum','mean'])
        #prev.round(1).to_csv('/home/pooja/PycharmProjects/datanalysis/featureEngeering/test.csv')
        a = prev.memory_usage(deep=True)
        b = prev.dtypes
        prev.columns = [str("_").join(col).strip() for col in prev.columns.values]

        prev= prev.drop(['TARGET' + "_" + f for f in ['min', 'max', 'sum']], axis=1)
        prev.rename(columns={'TARGET' + "_mean": 'TARGET'}, inplace=True)
        binned = binning(prev, 'TARGET', qCut=10, maxobjectFeatures=50, varCatConvert=1)




        ivData = iv_all(binned, 'TARGET', modeBinary=0)
        F=ivData.groupby('variable')['ivValue'].sum()
        N=0



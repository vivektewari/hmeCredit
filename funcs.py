import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import roc_curve, auc
from multiprocessing import Pool, Process, cpu_count, Manager
from iv import iv_all,binning
import time
import random
from os import path
import gc
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
def RMSE(dataset,actual,predicted):
    return (((dataset[actual]-dataset[predicted])**2).mean())**0.5
def logLoss(dataset,actual,predicted):
    dataset['error']=dataset[actual]*np.log(dataset[predicted]) + (1-dataset[actual])*np.log(1-dataset[predicted])
    return dataset
def deliquency(df1,rol,del_Var,monthVar, monthList=[12, 36, 120]):
    df = df1.copy()
    df_uniques = df.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'], ascending=False).drop_duplicates(
        subset=['SK_ID_PREV']) #taking latest information
    #df_uniques=df_uniques.drop(['NAME_CONTRACT_STATUS'], axis=1).set_index('SK_ID_PREV')
    #dfactives = df[df['NAME_CONTRACT_STATUS'] == "Active"].groupby(['SK_ID_PREV'])['NAME_CONTRACT_STATUS'].count()
    monthVarlist=[]
    for month in monthList:
        df["dpd_last_months_" + str(month)] = df.apply(
            lambda row: (1 + int(row[del_Var]) / 30) if (row[monthVar] > -month and row[del_Var] > 0) else 0,
            axis=1)
        monthVarlist.append("dpd_last_months_" + str(month))

    df2 = df.groupby(monthVarlist).max()
    final = df_uniques.join([df2])

    return final.reset_index()

def normalize(train,test):
        normalized_train=(train-train.mean())/train.std()
        normalized_test = (test - train.mean()) / train.std()
        return normalized_train,normalized_test

def aggregation(df1, rollupKey ='', monthVar='', aggFunc={}, monthList=[12, 36, 120]):#monh closeset is nbiggest
    df = df1.copy()

    df_uniques = df.sort_values([rollupKey, monthVar], ascending=False).drop_duplicates(
        subset=[rollupKey])  # taking latest information
    for month in monthList:
        df_temp = df[df[monthVar] > month]
        df_temp = df_temp.groupby([rollupKey]).agg(aggFunc)
        df_temp.columns = [str('_'+str(month)+"_").join(col).strip() for col in df_temp.columns.values]
        df_uniques=df_uniques.join(df_temp,on=[rollupKey])

    return df_uniques.fillna(0)
def crossVariable(df1,combination=None,target=None,varlist=None,ignoreList=None):
    if varlist==None:varlist=df1.columns
    elif ignoreList==None:varlist=list(set(df1.columns)-set(ignoreList))
    df=df1[[]]
    if combination is None:combs=combinations(varlist, 2)
    else:combs=combination
    for comb in combs:
        #print(comb[0] + "_" + comb[1] +"_c")
        df[comb[0] + "_&_" + comb[1] ] = df1[comb[0]] + "_&_"+df1[comb[1]]
        # try:
        #     #df[comb[0]+"_"+comb[1]+"m"]=df1[comb[0]]*df1[comb[1]]
        # except TypeError:
        #     #df[comb[0] + "_" + comb[1] + "m"] = df1[comb[0]] + df1[comb[1]]
        #
        # try:
        #     df[comb[0] + "_" + comb[1] + "d"] = df1[comb[0]] / df1[comb[1]]
        # except ZeroDivisionError:
        #     df[comb[0] + "_" + comb[1] + "d"]=np.nan
        # except TypeError:
        #     pass
    if target is not None:df[target]=df1[target]
    return df #.replace(np.inf,np.nan)
#low ram

def jugad(df1,combination,target,number,loc):
    temp=crossVariable(df1,combination,target=target,varlist=None,ignoreList=None)
    #temp['TARGET']=[random.randint(0, 1) for i in range(temp.shape[0])]
    ivData = iv_all(temp, target,modeBinary=0)
    #print("close")
    ivData.groupby('variable')['ivValue'].sum().to_csv(loc+str(number)+".csv",mode = 'a', header = False)

def crossVariablelowRam(df1,train,varlist=None,ignoreList=None,target=None,batch=10,loc=None):
    start = time.time()
    outputFile=pd.DataFrame(columns =['ivValue'])
    for i in range(batch):outputFile.to_csv(loc+str(i)+".csv")
    cores=cpu_count()
    pool = Pool(processes=cores)
    if varlist is None:varlist=df1.columns
    combs = combinations(varlist, 2)
    if ignoreList is not None:varlist=list(set(df1.columns)-set(ignoreList))

    coreNum=0
    excludes=[]
    binned = binning(df1,  qCut=10, maxobjectFeatures=50,varCatConvert=1,excludedList=excludes)
    varlist=list(set(varlist)-set(excludes))
    combs = combinations(varlist, 2)
    total_cross = len(list(combs))
    numberOfBatches = int(total_cross / batch)
    binned = binned.astype(str)
    binned=binned.join(train[[target]])
    #print(binned.dtypes)
    print(numberOfBatches)
    binned.columns=[ col.replace('n_',"").replace('c_',"") for col in binned.columns]
    combs = list(combinations(varlist, 2))
    i=0
    for i in range(0,16):
        cross=combs[i*batch:i*batch+batch]
        vars=list(set([com[0] for com in cross]).union(set([com[1] for com in cross])))+[target]
        pool.apply_async(jugad,args=(binned[vars],cross,target,i%batch,loc))
        print(i)
        coreNum=coreNum+1
        gc.collect()
    cross = combs[i * batch:total_cross]
    vars = list(set([com[0] for com in cross]).union(set([com[1] for com in cross]))) + [target]
    pool.apply_async(jugad, args=(binned[vars], cross, target, i % batch, loc))

    pool.close()
    pool.join()
    end = time.time()
    print(end - start)
#s_POS_CASH_balance=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets//POS_CASH_balance.csv")
data = pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets/train.csv")
target=data.set_index('SK_ID_CURR')[['TARGET']]
#data=data[['EXT_SOURCE_3','EXT_SOURCE_2','EXT_SOURCE_1','TARGET','SK_ID_CURR']]
#data['check']=1
#s_POS_CASH_balance=s_POS_CASH_balance.join(data[['SK_ID_CURR','TARGET']].st_index('SK_ID_CURR'),on='SK_ID_CURR')
x=crossVariablelowRam(data.drop('TARGET',axis=1).set_index('SK_ID_CURR'),target,ignoreList=['SK_ID_PREV','Unnamed: 0','SK_ID_CURR','TARGET'],target='TARGET',batch=10,loc="/home/pooja/PycharmProjects/datanalysis/featureEngeering/train")
#q=past(s_POS_CASH_balance,'SK_ID_CURR','MONTHS_BALANCE')

def isPrimaryKey(df,varList):
    """

    :param df:dataframe| for which we are checking primary key
    :param varList: list| varList by which primary key can be formed
    :return: boolean| True if yes
    """
    df['pk']=""
    for var in varList:
        df['pk']=df['pk']+df[var].map(str)

    return df.shape[0]==len(df['pk'].unique())

def lorenzCurve(y_test,y_score):
    n_classes = 1
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _= roc_curve(y_test, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #Plot of a ROC curve for a specific class

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()



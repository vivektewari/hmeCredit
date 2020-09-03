import numpy as np
from itertools import combinations
from sklearn.metrics import roc_curve, auc

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
def crossVariable(df1,varlist=None):
    if varlist==None:varlist=df1.columns
    df=df1[[]]
    combs=combinations(varlist, 2)
    for comb in combs:
        print(comb[0] + "_" + comb[1] + "m")
        try:
            df[comb[0]+"_"+comb[1]+"m"]=df1[comb[0]]*df1[comb[1]]
        except TypeError:
            df[comb[0] + "_" + comb[1] + "m"] = df1[comb[0]] + df1[comb[1]]


        try:
            df[comb[0] + "_" + comb[1] + "d"] = df1[comb[0]] / df1[comb[1]]
        except ZeroDivisionError:
            df[comb[0] + "_" + comb[1] + "d"]=np.nan

    return df.replace(np.inf,np.nan)
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



import pandas as pd
from dataExploration import distReports
import os
pk='SK_ID_CURR'
sLoc="/home/pooja/PycharmProjects/datanalysis/rawDatas/"
dLoc="/home/pooja/PycharmProjects/datanalysis/relevantDatasets/"
testLoc= dLoc + "test/" #change2
#folderChecks
if not os.path.exists(sLoc): raise Exception("inputFolderMissing:"+sLoc)
if not os.path.exists(testLoc): os.mkdir(testLoc)
if not os.path.exists(dLoc): os.mkdir(dLoc)


relevant=pd.read_csv(sLoc+'application_train.csv')
#relevant=relevant.sample()
#relevant.to_csv(dLoc+'test.csv')
relevant=relevant[[pk]]

datasetNames=['previous_application.csv','POS_CASH_balance.csv','installments_payments.csv','credit_card_balance.csv','bureau.csv']
final=pd.DataFrame()
for dataset in datasetNames:
    data=pd.read_csv(sLoc+dataset)
    data = data[data[pk].isin(relevant[pk])]
    #data.to_csv(dLoc+dataset)
    temp=distReports(data)
    temp['dataset']=dataset
    temp['floor']=579579579
    temp['cap']=579579579
    temp['mis']=579579579
    final=final.append(temp)
pk='SK_ID_BUREAU'
bureau=data[[pk]]
dataset='bureau_balance.csv'
data=pd.read_csv(sLoc+dataset)
data = data[data[pk].isin(bureau[pk])]
#data.to_csv(dLoc+'bureau_balance.csv')
temp = distReports(data)
temp['dataset'] = dataset
temp['floor'] = 579579579
temp['cap'] = 579579579
temp['mis'] = 579579579
final = final.append(temp)
final=final.reindex(['dataset','varName','floor','cap','mis']+list(set(final.columns)-set(['dataset','varName','floor','cap','mis'])),axis=1)
final.to_csv(testLoc + "allDescribe.csv")

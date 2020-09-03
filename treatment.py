import pandas as pd
import numpy as np
from dataExploration import distReports, treatment_outlier
import os
sLoc="/home/pooja/PycharmProjects/datanalysis/relevantDatasets/" #change1
dLoc=sLoc+"treated/" #change2
dLocT=dLoc+"test/" #change2

#folderChecks
if not os.path.exists(sLoc): raise Exception("inputFolderMissing:"+sLoc)
if not os.path.exists(dLoc): os.mkdir(dLoc)
if not os.path.exists(dLocT): os.mkdir(dLocT)


datasetNames=['POS_CASH_balance.csv','previous_application.csv','installments_payments.csv','credit_card_balance.csv','bureau.csv']
allTreatments = pd.read_csv(sLoc + "test/allDescribe.csv")
final=pd.DataFrame()
for dataset in datasetNames:
        data= pd.read_csv(sLoc + dataset)
        treatments=allTreatments[allTreatments['dataset']==dataset]
        vars=list(treatments['varName'])
        bottomCaps=list(treatments['floor'])
        upperCaps = list(treatments['cap'])
        temp=treatment_outlier(data, vars, bottomCaps, upperCaps, percentile=[0.01, 0.99],specialValue=579579579)
        temp.to_csv(dLoc+dataset)
        temp=distReports(temp)
        temp['dataset'] = dataset
        final = final.append(temp)
final=final.reindex(['dataset','varName']+list(set(final.columns)-set(['dataset','varName'])),axis=1)
final.to_csv(dLocT+'allDescribe.csv')


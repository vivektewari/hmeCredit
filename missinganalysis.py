import pandas as pd
import numpy as np
from dataExploration import distReports, treatment_outlier
dLoc="/home/pooja/PycharmProjects/datanalysis/relevantDatasets/"
datasetNames=['previous_application.csv','POS_CASH_balance.csv','installments_payments.csv','credit_card_balance.csv','bureau.csv']
data=pd.read_csv(dLoc+datasetNames[4]).fillna(579579579)

d=data[data['AMT_ANNUITY']==579579579]#[['CNT_INSTALMENT_MATURE_CUM','SK_ID_PREV']]
refData=pd.read_csv(dLoc+datasetNames[0]).fillna(579579579)
f=d.join(refData.set_index('SK_ID_PREV'),on='SK_ID_PREV')
#e=d.groupby(['NAME_CONTRACT_STATUS'])['NAME_CONTRACT_STATUS'].count()
c=0
import pandas as pd
from dataExploration import distReports,plotGrabh
import numpy as np
from iv import iv_all,binning
import matplotlib.pyplot as plt
from funcs import crossVariable
import time
import seaborn as sns
start = time.time()
rowsMax=10^10
dLoc="/home/pooja/PycharmProjects/datanalysis/finalDatasets/"
data=pd.read_csv('/home/pooja/PycharmProjects/datanalysis/finalDatasets/final.csv')

target='TARGET'

data['worstBucket']=data.apply(lambda row:max(0,row['worstBucketimonths:-1000_max'],row['curBucket_max']),axis=1)
data['total_annuity_active'] = data['AMT_ANNUITY_active_sum'] + data['AMT_ANNUITY_active_sumbur']
data['annuity_perc_income'] = data['total_annuity_active'] / data['AMT_INCOME_TOTAL']
#data['phone']=max(data['FLAG_EMP_PHONE'],data['FLAG_PHONE'])
data['home_dim_missing']=data.apply(lambda row:1 if row['NONLIVINGAREA_MODE']*row['NONLIVINGAREA_MODE'] is np.nan else 0,axis=1)
data['active']=data.apply(lambda row: row['active_sumbur']+row['activePos_sum']+row['activeCards_sum'],axis=1)
data=data.set_index('SK_ID_CURR')
# df1=crossVariable(data[['CNT_FAM_MEMBERS','ELEVATORS_MEDI','ENTRANCES_MEDI','FLOORSMAX_MEDI','FLOORSMAX_AVG','LIVINGAREA_AVG','TOTALAREA_MODE','YEARS_BEGINEXPLUATATION_AVG']])
# df2=crossVariable(data[["FLAG_DOCUMENT_" +str(i) for i in range(2,22)]])
# df3=crossVariable(data[['NAME_CONTRACT_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','NAME_INCOME_TYPE','NAME_PAYMENT_TYPE','NAME_TYPE_SUITE']])
# #a=plotGrabh(data,'TARGET',dLoc+"images/")
#df1=crossVariable(data[['active','not_approved_sum','AMT_GOODS_PRICE','AMT_INCOME_TOTAL','total_annuity_active','AMT_ANNUITY_active_sum']])
#data=data[[target,'annuity_perc_income','total_annuity_active','home_dim_missing','active']].join(other=[df1])
binned = binning(data, target)
# plt.figure(figsize = (15,10))
# sns.heatmap(binned .corr())
# plt.show()


ivData = iv_all(binned, target)
ivData.to_csv(dLoc + "iv_detailed.csv")
ivData.groupby('variable')['ivValue'].sum().to_csv(dLoc + "/" + "iv3.csv")
ivInfo = pd.read_csv(dLoc + "/" + "iv3.csv")
distRepo = distReports(data, ivInfo)
distRepo.to_csv(dLoc+ "/" + "summary.csv")
end = time.time()
print(end - start)
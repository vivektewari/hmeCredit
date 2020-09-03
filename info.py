import warnings
from dataExploration import distReports,plotGrabh
from iv import iv_all,binning
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
pk='SK_ID_CURR'
loc="/home/pooja/PycharmProjects/datanalysis/"
folder=['previous_application']
fileName=['previous_application.csv']
target='TARGET'
train=pd.read_csv(loc+"rawDatas"+"/application_train.csv")
for i in range(0,len(folder)):
    data=pd.read_csv(loc+folder[i]+"/"+fileName[i])
    data=data[data[pk].isin(train[pk])]
    data=data.join(train[[target,pk]].set_index(pk),on=[pk],how='left')
    #a = plotGrabh(data, target, loc + folder[i] + "/images/")

    if target not in data.columns:data[target]=0


    binned = binning(data, target)
    ivData = iv_all(binned, target)
    ivData.to_csv(loc +folder[i]+ "/"+"iv_detailed.csv")
    ivData.groupby('variable')['ivValue'].sum().to_csv(loc +folder[i]+"/"+ "iv3.csv")
    ivInfo = pd.read_csv(loc +folder[i]+"/"+ "iv3.csv")
    distRepo = distReports(data,ivInfo)
    distRepo.to_csv(loc +folder[i] +"/"+"summary.csv")
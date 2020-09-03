import pandas as pd
loc="/home/pooja/PycharmProjects/datanalysis/bureau/"
data=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/rawDatas/bureau.csv")
train=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/application_train/train.csv")
train2=train[['SK_ID_CURR','TARGET']]
bureauTrain=train2.set_index('SK_ID_CURR').join(data,on='SK_ID_CURR')
bureauTrain.to_csv(loc+"bur.csv")

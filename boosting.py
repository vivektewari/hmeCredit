from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
from funcs import lorenzCurve,normalize
from sklearn import metrics
train=pd.read_csv('/home/pooja/PycharmProjects/datanalysis/finalDatasets/final.csv')
test=pd.read_csv('/home/pooja/PycharmProjects/datanalysis/finalDatasets/finalTest.csv')
train['lowOccupation']=train['OCCUPATION_TYPE'].apply(lambda x: 1 if x in ['Cleaning staff','Cooking staff','Drivers','Laborers','Low-skill Laborers','Sales staff','Security staff','Waiters/barmen staff'] else 0)
train=train.replace([np.inf,np.nan,365243],0)#[train['active_sumbur']==0]
test['lowOccupation']=test['OCCUPATION_TYPE'].apply(lambda x: 1 if x in ['Cleaning staff','Cooking staff','Drivers','Laborers','Low-skill Laborers','Sales staff','Security staff','Waiters/barmen staff'] else 0)
test=test.replace([np.inf,np.nan,365243],0)#[train['active_sumbur']==0]
test['TARGET']=0
test=test.set_index('SK_ID_CURR')
validation =0
if validation ==1 :
    test=train.sample(n=10000)
    train=train.drop(test.index,axis=0)
train_y=train['TARGET']
test_y=test['TARGET']


varSelected=['EXT_SOURCE_3','EXT_SOURCE_2','EXT_SOURCE_1','DAYS_EMPLOYED','AMT_GOODS_PRICE','DAYS_CREDIT_min','PRODUCT_d_mean','REGION_RATING_CLIENT_W_CITY','lowOccupation','utilization_-6_mean_max','payTominDue_-6_mean_mean','EXT_SOURCE_3','EXT_SOURCE_2','EXT_SOURCE_1','utilization_-6_mean_max','payTominDue_-6_mean_mean']#,'active_sumbur','active_mean'
#train,test=normalize(train[varSelected],test[varSelected])
train,test=normalize(train[varSelected],test[varSelected])
train['TARGET']=train_y
test['TARGET']=test_y
train.describe().to_csv("/home/pooja/PycharmProjects/datanalysis/finalDatasets/des.csv")









abc = AdaBoostClassifier(n_estimators=200,
                         learning_rate=1)
# Train Adaboost Classifer
mlp = abc.fit(train[varSelected], train['TARGET'])

#Predict the response for test dataset
trainer = pd.DataFrame(mlp.predict_proba(train[varSelected].values), columns=['good', 'TARGET'], index=train.index)[
    ['TARGET']]
submision = pd.DataFrame(mlp.predict_proba(test[varSelected].values), columns=['good', 'TARGET'], index=test.index)[
    ['TARGET']]
print(train[['TARGET']].mean(axis=0), trainer[['TARGET']].mean(axis=0))
lorenzCurve(train['TARGET'].values.flatten(),trainer["TARGET"].values.flatten())
submision.to_csv("/home/pooja/PycharmProjects/datanalysis/finalDatasets/submission.csv")
score_test = metrics.roc_auc_score(test['TARGET'], submision[['TARGET']])
score_train = metrics.roc_auc_score(train['TARGET'], trainer[['TARGET']])
print(score_test,score_train)
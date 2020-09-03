from sklearn.preprocessing import StandardScaler
import warnings
import time
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
from funcs import lorenzCurve,normalize,logLoss
from statsmodels.tools.tools import add_constant
start = time.time()
warnings.filterwarnings("ignore")
from multiprocessing import Pool, Process, cpu_count, Manager
from sklearn import metrics
train=pd.read_csv('/home/pooja/PycharmProjects/datanalysis/finalDatasets/final.csv')
test=pd.read_csv('/home/pooja/PycharmProjects/datanalysis/finalDatasets/finalTest.csv')
rawTest=test
train['lowOccupation']=train['OCCUPATION_TYPE'].apply(lambda x: 1 if x in ['Cleaning staff','Cooking staff','Drivers','Laborers','Low-skill Laborers','Sales staff','Security staff','Waiters/barmen staff'] else 0)
train=train.replace([np.inf,np.nan,365243],0)#[train['active_sumbur']==0]
test['lowOccupation']=test['OCCUPATION_TYPE'].apply(lambda x: 1 if x in ['Cleaning staff','Cooking staff','Drivers','Laborers','Low-skill Laborers','Sales staff','Security staff','Waiters/barmen staff'] else 0)
test=test.replace([np.inf,np.nan,365243],0)#[train['active_sumbur']==0]
test['TARGET']=0
test=test.set_index('SK_ID_CURR')
validation =1
if validation ==1 :
    test=train.sample(n=10000)
    train=train.drop(test.index,axis=0)
    rawTest = test
train_y=train['TARGET']
test_y=test['TARGET']


varSelected=['DAYS_EMPLOYED','AMT_GOODS_PRICE','DAYS_CREDIT_min','PRODUCT_d_mean','REGION_RATING_CLIENT_W_CITY','lowOccupation','utilization_-6_mean_max','payTominDue_-6_mean_mean','EXT_SOURCE_3','EXT_SOURCE_2','EXT_SOURCE_1','utilization_-6_mean_max','payTominDue_-6_mean_mean']#,'active_sumbur','active_mean'
train,test=normalize(train[varSelected],test[varSelected])
train['TARGET']=train_y
test['TARGET']=test_y
#X = train[varSelected].assign(const=1)
#X=X.replace([np.inf,np.nan],0)
#d=X.isna().sum()
#c=pd.Series([variance_inflation_factor(X.values, i)for i in range(X.shape[1])],index=X.columns)
f=0

def uplevel(mlp, train, trainTarget, i, j):
    mlp.fit(train, trainTarget["TARGET"])
    return [mlp, i, j]


def jugad(que, mlp, train, trainTarget, i, j):
    que.put(uplevel(mlp, train, trainTarget, i, j))


def runModel(train, test, trainTarget, testTarget, targetVar='TARGET', subMode=None,submission=0):
    print("started")
    from sklearn.neural_network import MLPClassifier
    if subMode is None:
        que = Manager().Queue()
        pool = Pool(processes=1)
        counter = 0

    dict1 = {}
    L1 = [i for i in range(11, 16)]
    L2 = [i for i in range(4, 9)]
    for i in L1:
        for j in L2:
            if j < i and subMode is None:
                if j != 0:
                    mlp = MLPClassifier(hidden_layer_sizes=(i, j), max_iter=250, alpha=1e-4,
                                        solver='sgd', verbose=False, tol=1e-4, random_state=1,
                                        learning_rate_init=0.1)
                else:
                    mlp = MLPClassifier(hidden_layer_sizes=(i), max_iter=250, alpha=1e-4,
                                        solver='sgd', verbose=False, tol=1e-4, random_state=1,
                                        learning_rate_init=0.1)
                pool.apply_async(jugad, args=(que, mlp, train, trainTarget, i, j))
                counter = counter + 1
            elif subMode is not None:
                mlp = MLPClassifier(hidden_layer_sizes=subMode, max_iter=250, alpha=1e-4,
                                    solver='sgd', verbose=False, tol=1e-4, random_state=1,
                                    learning_rate_init=0.1)
                mlp.fit(train, trainTarget["TARGET"])
                trainer = pd.DataFrame(mlp.predict_proba(train.values), columns=['good', 'TARGET'], index=train.index)[
                    ['TARGET']]
                submision = pd.DataFrame(mlp.predict_proba(test.values), columns=['good', 'TARGET'], index=test.index)[
                    ['TARGET']]
                print(trainTarget[['TARGET']].mean(axis=0), trainer[['TARGET']].mean(axis=0))
                #lorenzCurve(trainTarget["TARGET"].values.flatten(),trainer['TARGET'].values.flatten())
                rawTest['actual'] = testTarget['TARGET']
                rawTest['TARGET1']=submision['TARGET']
                error = logLoss(rawTest[varSelected+['actual', 'TARGET1']], 'actual', 'TARGET1').sort_values(['error'])
                error.to_csv("/home/pooja/PycharmProjects/datanalysis/finalDatasets/error.csv")
                return pd.DataFrame(mlp.predict_proba(test.values), columns=['good','TARGET'],
                                    index=testTarget.index)[['TARGET']]

    pool.close()
    pool.join()
    for element in range(counter):
        field = que.get()
        mlp = field[0]
        i = field[1]
        j = field[2]
        trainer = pd.DataFrame(mlp.predict_proba(train.values), columns=['good', 'TARGET'], index=train.index)[
            ['TARGET']]
        submision = pd.DataFrame(mlp.predict_proba(test.values), columns=['good', 'TARGET'], index=test.index)[
            ['TARGET']]
        # submision.index.name='SK_ID_CURR'
        # print(submision.shape)
        # submision.to_csv("submission.csv")
        score_test = metrics.roc_auc_score(testTarget['TARGET'], submision[['TARGET']])
        score_train = metrics.roc_auc_score(trainTarget['TARGET'], trainer[['TARGET']])



        try:
            dict1[str(i) + "_" + str(j)] = str(score_train) + "_" + str(score_test)
        except:
            print(i, j)
            pass

    print("starting")
    for key in dict1.keys():
        print(key)
        print(dict1[key])
    return None
subCard=runModel(train[varSelected].replace([np.inf,np.nan],0),test[varSelected].replace([np.inf,np.nan],0),train[['TARGET']],test[['TARGET']],subMode=(14,7))#,subMode=(15,5)
#subCard=runModel(train[varSelected].replace([np.inf,np.nan],0),test[varSelected].replace([np.inf,np.nan],0),train[['TARGET']],test[['TARGET']])#,subMode=(15,5)

#subCard=runModel(train_t,test_t,train[['TARGET']],test[['TARGET']])#,subMode=(15,5)
subCard.to_csv("/home/pooja/PycharmProjects/datanalysis/finalDatasets/submission.csv")
end = time.time()

print(end - start)
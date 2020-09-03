import pandas as pd
from sklearn.model_selection import train_test_split
from dataExploration import distReports

data=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/rawDatas/application_train.csv")
c=distReports(data)
v=0

X=data.drop("TARGET",axis=1)
y=data[["TARGET"]]
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.7, test_size=0.3,
                                                      random_state=0)
trainall=X_train.join(y_train)
testall=X_valid.join(y_valid)
trainall.to_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets/train.csv")
testall.to_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets/valid.csv")

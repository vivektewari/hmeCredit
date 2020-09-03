import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
train=pd.read_csv('/home/pooja/PycharmProjects/datanalysis/finalDatasets/final.csv')
test=pd.read_csv('/home/pooja/PycharmProjects/datanalysis/finalDatasets/finalTest.csv')
train['lowOccupation']=train['OCCUPATION_TYPE'].apply(lambda x: 1 if x in ['Cleaning staff','Cooking staff','Drivers','Laborers','Low-skill Laborers','Sales staff','Security staff','Waiters/barmen staff'] else 0)
train=train.replace([np.inf,np.nan,365243],0)#[train['active_sumbur']==0]
test['lowOccupation']=test['OCCUPATION_TYPE'].apply(lambda x: 1 if x in ['Cleaning staff','Cooking staff','Drivers','Laborers','Low-skill Laborers','Sales staff','Security staff','Waiters/barmen staff'] else 0)
test=test.replace([np.inf,np.nan,365243],0)#[train['active_sumbur']==0]
test['TARGET']=0
test=test.set_index('SK_ID_CURR')
test=train.sample(n=10000)
train=train.drop(test.index,axis=0)

varSelected=['DAYS_EMPLOYED','AMT_GOODS_PRICE','DAYS_CREDIT_min','PRODUCT_d_mean','REGION_RATING_CLIENT_W_CITY','lowOccupation','utilization_-6_mean_max','payTominDue_-6_mean_mean','utilization_-6_mean_max','payTominDue_-6_mean_mean']#,'active_sumbur','active_mean',,'EXT_SOURCE_3','EXT_SOURCE_2','EXT_SOURCE_1'
#X = train[varSelected].assign(const=1)

clf = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)

# Train Decision Tree Classifer
clf = clf.fit(train[varSelected],train['TARGET'])

#Predict the response for test dataset
y_pred = clf.predict_proba(train[varSelected])
train['predicted']=y_pred[:,1]
#score_test = metrics.roc_auc_score(testTarget['TARGET'], submision[['TARGET']])
score_train = metrics.roc_auc_score(train['TARGET'],train['predicted'])
print(score_train)

from sklearn import tree
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf,
               feature_names = varSelected,
               class_names='TARGET',
               filled = True);
fig.savefig('/home/pooja/PycharmProjects/datanalysis/finalDatasets/imagename.png')
#dot_data = StringIO()
# export_graphviz(clf)#, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True,feature_names = varSelected,class_names=['0','1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('/home/pooja/Downloads/tree.png')
# Image(graph.create_png())
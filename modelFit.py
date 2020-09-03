import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
data=pd.read_csv("/home/pooja/Downloads/crossCell.csv")
data['hd_occupationType']=data['occupationType'].apply(lambda x: 1 if  x in ['Cleaning staff','Cooking staff','Drivers','Laborers','Low-skill Laborers'] else 0)
data['REGION_RATING_CLIENT_W_CITY']=data['REGION_RATING_CLIENT_W_CITY']-2

feature_cols = ['OCCUPATION_TYPE', 'Term', 'DAYS_EMPLOYED','AMT_GOODS_PRICE','Employment_Status','Income_Range','Time_with_Bank','Value_of_Property']
feature_cols = ['Credit_Score','Income_Range','Mosaic_Class']

X = source[feature_cols]# Features
s = (X.dtypes == 'object')
object_cols = list(s[s].index)
from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data


# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    X[col] = label_encoder.fit_transform(X[col])

y = source.PPI# Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

clf = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import log_loss

loss=log_loss(y_test, y_pred, eps=1e-15)
print(loss)
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('/home/pooja/Downloads/tree.png')
Image(graph.create_png())
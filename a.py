
import pandas as pd
import matplotlib.pyplot as plt



application_train=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/application_train.csv")
temp = application_train["NAME_EDUCATION_TYPE"]
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
application_train['NAME_EDUCATION_TYPE'].value_counts().plot(kind='pie',autopct='%1.1f%%',
        pctdistance=0.9, labeldistance=0.2,radius=1.5)
plt.show()
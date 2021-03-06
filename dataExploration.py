import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numbers
import matplotlib.pyplot as plt

def treatment_outlier(df, vars, bottomCaps=None, upperCaps=None, percentile=[0.01, 0.99],specialValue=np.nan):
        """
        :param df: dataframe|Dataset for which outlier treatment been done.
        :param vars: string|Variable for which outlier treatment will be done
        :param bottomCaps:string(method) or number|floored value or method
        :param upperCaps:string(method) or number|capped value or method
        :param percentile:decimal 0 to 1 list(2)|which percecentile value is used for caped or flored
        :return: dataframe|with variable been replaced by outlier treated variable. No variable change
        """

        for i in range(0,len(vars)):
                circuit=[np.nan,np.nan] #upper and lower bound collectid in list
                caps=[bottomCaps,upperCaps]#upper and lower bound parameters in list
                for j in range(0,2):
                        #f=caps[j][i]
                        if caps[j] is not None:
                                if caps[j][i]=='percentile':
                                        percentile1 = df[vars[i]].quantile(percentile[j])
                                        circuit[j]=percentile1
                                elif caps[j][i]in [specialValue,str(specialValue)]:
                                        circuit[j] = np.nan
                                elif caps[j][i]=='nan':
                                        circuit[j]=np.nan
                                else:
                                        circuit[j]=float(caps[j][i])
                #print(vars[i], circuit[1])
                if not circuit[0] in [np.nan,float('nan')] :  df[vars[i]][df[vars[i]]<circuit[0]]   =    circuit[0]
                if not circuit[1] in [np.nan,float('nan')] :  df[vars[i]][df[vars[i]] > circuit[1]] =    circuit[1]

        return df



def distReports(df,ivReport=None):
        mis = pd.DataFrame({'varName':df.columns.values,'missing':df.isnull().values.sum(axis=0)}, index=df.columns.values)  # new df from existing
        basta = (df.describe()).transpose()
        uniques=pd.DataFrame({'nuniques':df.nunique()},index=df.columns.values)
        mis['missing_percent'] = mis['missing'] / df.shape[0]  # new column creation
        final = mis.join(basta).join(uniques) # join using index
        final['uniqueValues']=final['varName'].apply(lambda x:df[x].unique()[0:50])
        if ivReport is not None :final.join(ivReport)
        return final
import numpy as np
import matplotlib.pyplot as plt
def plotGrabh(df,target,location):
        defaults = df[df[target] == 1]
        objectCols = list(df.select_dtypes(include=['object']).columns)
        allCols = df.columns
        numCols = list(set(allCols) - set(objectCols) - set([target]))

        uniques = pd.DataFrame({'nuniques': df[numCols].nunique()}, index=df[numCols].columns.values)
        numCats=list(uniques[uniques['nuniques']<10].index)
        catCols = objectCols + numCats
        contCols=list(set(allCols)-set(catCols))



        for col in catCols:
                fig = plt.figure()
                temp=df[[col]].fillna('Missing')
                temp2 = defaults[[col]].fillna('Missing')
                ax0 = fig.add_subplot(121)
                temp[col].value_counts().plot(kind='pie', autopct='%1.1f%%',
                                                                     pctdistance=0.9, labeldistance=1.2, radius=1.5)
                ax0 = fig.add_subplot(122)
                temp2[col].value_counts().plot(kind='pie', autopct='%1.1f%%',
                                      pctdistance=0.9, labeldistance=1.2, radius=1)
                plt.savefig(location+col+'.png')
        #df[target]=df[target].astype('category')
        for col in contCols:
                fig = plt.figure(figsize=(8, 8))
                x=df[df[target]==0][col]
                d=df[df[target]==1][col]
                sns.kdeplot(df[col], label="all")
                sns.kdeplot(x,  label="0")
                sns.kdeplot(d,  label="1")
                #ax0 = fig.add_subplot(1,3,2)
                plt.savefig(location+col+'.png')
                fig = plt.figure(figsize=(8, 8))
                sns.set(style="whitegrid")
                ax = sns.boxplot(x=col,hue_order=target,data=df)
                #ax0 = fig.add_subplot(3, 3, 2)
                plt.savefig(location + col + '_blot.png')





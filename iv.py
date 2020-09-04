import pandas as pd
import numpy as np

def calculate_woe_iv(dataset, feature, target):
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        })

    dset = pd.DataFrame(lst)

    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()

    dset = dset.sort_values(by='WoE')

    return dset, iv


from sklearn.preprocessing import OneHotEncoder


def binning(df, target=None, qCut=10, maxobjectFeatures=50,varCatConvert=0,excludedList=[]):
    output = pd.DataFrame(index=df.index, columns=[])

    objectCols = list(df.select_dtypes(include=['object']).columns)
    allCols = df.columns
    numCols = set(allCols) - set(objectCols)
    if target is not None:numCols-set([target])
    uniques = pd.DataFrame({'nuniques': df[numCols].nunique()}, index=df[numCols].columns.values)
    numCats = list(uniques[uniques['nuniques'] < 50].index)
    catCols = objectCols + numCats
    contCols = list(set(allCols) - set(catCols))
    for feature in contCols:
        temp = df[[feature]]
        #print(feature)
        missings = temp[temp.isnull().any(axis=1)]  # 'getting missing dataset'
        missings["n_" + feature] = 'Missing'

        temp = temp.drop(missings.index, axis=0)  # 'non missing dataset'

        try:
            temp["n_" + feature] = pd.qcut(temp[feature], q=qCut, duplicates='drop')
        except IndexError:
            dices = min(df[feature].nunique(), qCut)
            if dices != 0 and temp.shape[0] > 0:
                temp["n_" + feature] = pd.qcut(temp[feature], q=dices, duplicates='drop')
        # print(temp.shape[0])
        output = output.join(temp.drop(feature, axis=1).append(missings.drop(feature, axis=1)))
        # print(output.shape[0])
    for feature in catCols:
        # print(feature)
        temp = df[[feature]]
        temp = temp.fillna('Missing')
        temp["c_" + feature] = temp[feature]
        if temp[feature].nunique() > maxobjectFeatures:
            excludedList.append(feature)
            print("removed as too many categories:" + feature)
        else:
            output = output.join(temp.drop(feature, axis=1))
    if varCatConvert==1: return output
    for col in output.columns:
        # X[col] = X[col].astype('category',copy=False)
        dummies = pd.get_dummies(output[col], prefix=col + '___')
        output = pd.concat([output, dummies], axis=1)
        output = output.drop(col, axis=1)
    if target is not None:output = output.join(df[[target]])
    return output


def iv_all(X,target,modeBinary=1):
    rowCount=X.shape[0]

    ivData=pd.DataFrame(index=X.columns,columns=['ivValue','WoE','Dist_Good','Dist_Bad','%popuation','badRate','variable'])
    for col in X.columns:
        if col == target: continue
        else:
            #print('WoE and IV for column: {}'.format(col))
            df, iv = calculate_woe_iv(X[[col,target]], col, target)
            if modeBinary==0:
                ivData['ivValue'][col] = iv
                ivData['variable'][col] = col.split('___')[0]

            else:
                df.index=df.Value
                #print(df)
                ivData['ivValue'][col]=df['IV'][1]
                ivData['Dist_Good'][col] = df['Distr_Good'][1]
                ivData['Dist_Bad'][col] = df['Distr_Bad'][1]
                ivData['WoE'][col] = df['WoE'][1]
                ivData['badRate'][col] = df['Bad'][1] / float(df['All'][1])
                ivData['%popuation'][col] = df['All'][1]/rowCount
                ivData['variable'][col]=col.split('___')[0]

    return ivData

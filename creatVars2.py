import pandas as pd
from iv import binning
def getVariables(fullString):
    operator1 = None
    aggregator1 = None
    varList=fullString.split("_&_")
    varCount = 0
    for var in varList:
        if var[0:2] in ['c_','n_']:var=var.replace('c_',"").replace('n_',"")
        for operat in ['m_','d_',"_"]:
            for aggregator in ['sum','mean','max','min']:
                searchText=operat+aggregator
                s=var.find(searchText)
                if s >0:
                    var=var.replace(searchText,"")
                    operator1=operat
                    aggregator1=aggregator
                    break

        varList[varCount]=var
        varCount=varCount+1
    if len(varList)==1:varList.append(None)
    if operator1 is None:operator1="j_"
    return {'varCount':len(varList),'vars':[varList[0],varList[1]],'operator':operator1,'aggregator':aggregator1}

#c=getVariables('EXT_SOURCE_3_&_EXT_SOURCE_2')
#c=getVariables('n_AMT_CREDIT_LIMIT_ACTUALd_mean')
def getVars():
    file=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/feature_cross/allIV.csv")
    f=file[file['select']==1]
    #'application_train.csv',
    main = pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets/application_train.csv")
    #excludes=[]
    #main=main.set_index('SK_ID_CURR')
    #binned = binning(main, qCut=10, maxobjectFeatures=50, varCatConvert=1, excludedList=excludes)
    #binned = binned.drop('TARGET', axis=1)
    #binned = binned.astype(str)
    #t=binned.dtypes
    #binned = binned.reset_index()
    # binned.columns = [col.replace('n_', "").replace('c_', "") for col in binned.columns]
    #
    # binned.to_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets/trainBinned.csv")
    datasetNames = ['trainBinned.csv','POS_CASH_balance.csv', 'previous_application.csv', 'installments_payments.csv',
                    'credit_card_balance.csv', 'bureau.csv']
    main=main.set_index('SK_ID_CURR')[['TARGET']]
    for datasetName in datasetNames:
        temp=f[f['dataset']==datasetName]
        vars=temp['varName'].to_list()
        s=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets/"+datasetName)
        d1=s[['SK_ID_CURR']]
        output=pd.DataFrame(index=d1.groupby('SK_ID_CURR')['SK_ID_CURR'].mean())
        d1=d1.set_index('SK_ID_CURR')
        s=s.set_index('SK_ID_CURR')
        for var in vars:
            t1=getVariables(var)
            print(var)
            if t1['vars'][1] is not None:
                if t1['operator']=='m_' :
                    d1[t1['vars'][0]+ "_&_" + t1['vars'][1]+'m_']=s[t1['vars'][0]]*s[t1['vars'][1]]
                elif t1['operator']=='d_' :
                    d1[t1['vars'][0]+ "_&_" + t1['vars'][1]+'d_']=s[t1['vars'][0]]/s[t1['vars'][1]]
                elif t1['operator']=='j_' :
                    try:
                        d1[t1['vars'][0]+ "_&_" + t1['vars'][1]+'j_']=s[t1['vars'][0]] +"_&_"+ s[t1['vars'][1]]
                    except:
                        print("cant'process:" +var)

                else:pass
            if t1['aggregator'] is not None:
                if t1['vars'][1] is not None:
                    output[t1['vars'][0]+ "_&_" + t1['vars'][1]+t1['aggregator']]=d1.groupby('SK_ID_CURR')[t1['vars'][0]+ "_&_" + t1['vars'][1]+t1['operator']].agg([t1['aggregator'] ]).apply(list)
                else:
                    output[t1['vars'][0]+"_"+t1['aggregator']  ] = s.groupby('SK_ID_CURR')[t1['vars'][0]].agg([t1['aggregator']]).apply(list)
            else:
                try:
                    output[t1['vars'][0]+ "_&_" + t1['vars'][1]+'j_'] =  d1[t1['vars'][0]+ "_&_" + t1['vars'][1]+'j_']
                except:
                    print("cant'process:" + var)
        output[datasetName]=1
        main=main.join(output)
    return main

#e=getVars()
#e.to_csv("/home/pooja/PycharmProjects/datanalysis/feature_cross/crossDataset.csv")

old=pd.read_csv('/home/pooja/PycharmProjects/datanalysis/finalDatasets/final.csv')
new=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/feature_cross/crossDataset.csv")
combined=old.set_index('SK_ID_CURR').join(new.set_index('SK_ID_CURR').drop(['TARGET', 'DAYS_CREDIT_min'],axis=1))
combined.to_csv('/home/pooja/PycharmProjects/datanalysis/finalDatasets/final_withCross.csv')
c=0








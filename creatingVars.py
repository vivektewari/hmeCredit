from funcs import aggregation,isPrimaryKey
import pandas as pd
import numpy as np
import os
from dataExploration import distReports
all=True
#dLoc="/home/pooja/PycharmProjects/datanalysis/relevantDatasets/"
sLoc="/home/pooja/PycharmProjects/datanalysis/relevantDatasets/treated/"
dLoc=sLoc+"final/" #change2
dLocT=dLoc+"test/" #change2

#folderChecks
if not os.path.exists(sLoc): raise Exception("inputFolderMissing:"+sLoc)
if not os.path.exists(dLoc): os.mkdir(dLoc)
if not os.path.exists(dLocT): os.mkdir(dLocT)
if all:# creating var for pos balance and rolling up to prev_id
    data=pd.read_csv(sLoc+'POS_CASH_balance.csv')
    #a=isPrimaryKey(data,['SK_ID_PREV','MONTHS_BALANCE']) #check of primary key
    data['bucket'] = data['SK_DPD_DEF'].apply(lambda row: (1 + int(row / 30)) if  row> 0 else 0)
    a=aggregation(data,'SK_ID_PREV','MONTHS_BALANCE',{'bucket':['max'],'MONTHS_BALANCE':['min','max']},[-3,-6,-12,-1000])
    a['pos'] = 1
    temp = distReports(a)
    temp['dataset'] = 'POS_CASH_balance.csv'
    temp.to_csv(dLocT+'allDescribe.csv')

    #a['active'] = a['NAME_CONTRACT_STATUS'].apply(lambda x: 1 if x == 'Active' else 0)
if all:# creating var for credit card and rolling up to prev_id
    data = pd.read_csv(sLoc + 'credit_card_balance.csv')
    # a=isPrimaryKey(data,['SK_ID_PREV','MONTHS_BALANCE']) #check of primary key
    data['utilization'] = data['AMT_BALANCE'] / data['AMT_CREDIT_LIMIT_ACTUAL']
    data['bucket'] = data['SK_DPD_DEF'].apply(lambda row: (1 + int(row / 30)) if row > 0 else 0)
    data['payTominDue'] = data['AMT_PAYMENT_TOTAL_CURRENT'] / data['AMT_INST_MIN_REGULARITY']
    b = aggregation(data.replace([np.inf, -np.inf], 0), 'SK_ID_PREV', 'MONTHS_BALANCE',
                    {'bucket': ['max'], 'utilization': ['max', 'mean'], 'payTominDue': ['max', 'mean'],
                     'MONTHS_BALANCE': ['min', 'max']}, [-3, -6, -12, -1000])
    b['cards'] = 1
    b['active'] = b['NAME_CONTRACT_STATUS'].apply(lambda x: 1 if x == 'Active' else 0)
    temp = distReports(b)
    report = pd.read_csv(dLocT + 'allDescribe.csv')
    temp['dataset'] = 'credit_card_balance.csv'
    report.append(temp).to_csv(dLocT + 'allDescribe.csv')

if all:#merging dataset with prev application
    data = pd.read_csv(sLoc + 'previous_application.csv')
    data=data.set_index('SK_ID_PREV')
    data=data.join(other=a.set_index('SK_ID_PREV'),rsuffix='pos')
    data = data.join(other=b.set_index('SK_ID_PREV'),rsuffix='card')
    temp = distReports(data)
    report=pd.read_csv(dLocT+'allDescribe.csv')
    temp['dataset'] = 'mergedPrevppl.csv'
    report.append(temp).to_csv(dLocT+'allDescribe.csv')
if all:# prev aplication roll up to sk_id_cur
    data['tenureLeft']=data['CNT_INSTALMENT_FUTURE']/data['CNT_INSTALMENT']
    data['tenureExceded']=data.apply(lambda row:row['DAYS_LAST_DUE']/float(row['DAYS_TERMINATION']) if row['DAYS_LAST_DUE']!=365243 else np.nan,axis=1)
    data['activeCards']=data['cards']*data['active']
    data['activePos']=data['pos']*data['active']
    data['AMT_ANNUITY_active'] = data['AMT_ANNUITY'] * data['active']
    data['AMT_APPLICATION_approved']=data['AMT_APPLICATION']/data['AMT_CREDIT'] #latest
    data['down_payment_made'] = data['AMT_DOWN_PAYMENT'] / data['AMT_GOODS_PRICE']  # latest
    data['not_approved'] =data['NAME_CONTRACT_STATUS'].apply(lambda row: 0 if row=='Approved' else 1)
    data['REJECT_REASON_NOT_XAP']=data['CODE_REJECT_REASON'].apply(lambda row: 0 if row=='XAP' else 1)
    data['yieldGroup_d'] = data['NAME_YIELD_GROUP'].apply(lambda row: 0 if row in ['low_action', 'low_normal'] else 1)
    #RATE_DOWN_PAYMENT,RATE_INTEREST_PRIMARY,RATE_INTEREST_PRIVILEGED,NAME_PAYMENT_TYPE,CNT_PAYMENT,NFLAG_INSURED_ON_APPROVAL
    data['PRODUCT_d']=data['PRODUCT_COMBINATION'].apply(lambda row: 0 if row in ['Cash X-Sell: low','POS household without interest','POS industry with interest','POS industry without interest','POS industry with interest','POS industry without interest'] else 1)
    #RATE_DOWN_PAYMENT,RATE_INTEREST_PRIMARY,RATE_INTEREST_PRIVILEGED,NAME_PAYMENT_TYPE,CNT_PAYMENT,NFLAG_INSURED_ON_APPROVAL
    data_latest=data.sort_values(['SK_ID_CURR','DAYS_FIRST_DRAWING']).reset_index()[['SK_ID_CURR','DAYS_FIRST_DRAWING','RATE_DOWN_PAYMENT','RATE_INTEREST_PRIMARY','RATE_INTEREST_PRIVILEGED','NAME_PAYMENT_TYPE','CNT_PAYMENT','NFLAG_INSURED_ON_APPROVAL']].drop_duplicates(subset=['SK_ID_CURR'])
    for i in [-3,-6,-12,-1000]:
        data['worstBucketimonths:'+str(i)]=data.apply(lambda row:max(0,row['bucket_'+str(i)+"_max"],row['bucket_'+str(i)+"_maxcard"]),axis=1)
    data = data.groupby(['SK_ID_CURR']).agg(
        {'payTominDue_-3_mean':['mean','min'],'payTominDue_-6_mean':['mean','min'],'payTominDue_-12_mean':['mean','min'],'payTominDue_-1000_mean':['mean','min'],'worstBucketimonths:-3':['sum','max'],'worstBucketimonths:-6':['sum','max'],'worstBucketimonths:-12':['sum','max'],'worstBucketimonths:-1000':['sum','max'],'utilization_-3_mean':['sum'],'utilization_-6_mean':['max'],'utilization_-12_mean':['max'],'utilization_-1000_mean':['max'],'active': ['sum', 'count'], 'activeCards': ['sum'], 'activePos': ['sum'], 'AMT_ANNUITY_active': ['sum'],'AMT_APPLICATION_approved':['sum'],
         'tenureExceded':['mean','max'],'not_approved': ['sum','mean'],'down_payment_made':['mean'], 'REJECT_REASON_NOT_XAP':['sum'],'PRODUCT_d': ['sum','mean'],'yieldGroup_d': ['sum'],'DAYS_FIRST_DRAWING':['min'],'MONTHS_BALANCE_-1000_maxcard':['min'],'MONTHS_BALANCE_-1000_max':['min'],
         'NFLAG_INSURED_ON_APPROVAL':['mean','max'],'tenureLeft':['mean','max'],'CNT_PAYMENT':['mean','max']})
    data.columns = [str("_").join(col).strip() for col in data.columns.values]
    d=data.join( data_latest.set_index('SK_ID_CURR'))
    d['MOB']=d.apply(lambda row:min(row['MONTHS_BALANCE_-1000_max_min'],row['MONTHS_BALANCE_-1000_maxcard_min']),axis=1)
if all:  # bureau_bal roll up to sk_id_bur
        data = pd.read_csv(sLoc + 'bureau_balance.csv')
        data['bucket'] = data['STATUS'].apply(lambda row: int(row) if row not in ['C','X']  else -1)

        e = aggregation(data, 'SK_ID_BUREAU', 'MONTHS_BALANCE', {'bucket': ['max','min'],'MONTHS_BALANCE': ['min']},[-3, -6, -12, -1000])
        e['bureau'] = 1
        e['bureauMOB'] = e['MONTHS_BALANCE_-1000_min']
        temp = distReports(e)
        report = pd.read_csv(dLocT + 'allDescribe.csv')
        temp['dataset'] = 'bureau_balance.csv'
        report.append(temp).to_csv(dLocT + 'allDescribe.csv')
if all:  # burea roll up to sk_id_curr
        data = pd.read_csv(sLoc + 'bureau.csv')
        data = data.set_index('SK_ID_BUREAU')
        data = data.join(other=e.set_index('SK_ID_BUREAU'), rsuffix='bureau')
        data['curBucket']=data['CREDIT_DAY_OVERDUE'].apply(lambda row: (1 + int(row / 30)) if row > 0 else 0)
        data['active'] = data['CREDIT_ACTIVE'].apply(lambda x: 1 if x =='Active' else 0)
        data['AMT_ANNUITY_active'] = data['AMT_ANNUITY'] * data['active']
        data['overDue_perc']=data['AMT_CREDIT_MAX_OVERDUE']/data['AMT_CREDIT_SUM']
        data['bureauMOB'].fillna(0,inplace=True)

        f = data.groupby('SK_ID_CURR').agg(
            {'overDue_perc':['sum','mean','max'],'active': ['sum', 'count','mean'], 'DAYS_CREDIT': ['min'],
             'curBucket': ['max','mean'], 'DAYS_CREDIT_ENDDATE': ['max'],'CNT_CREDIT_PROLONG':['sum'],'AMT_CREDIT_SUM':['sum'],
             'AMT_CREDIT_SUM_DEBT':['sum'],'AMT_CREDIT_SUM_LIMIT':['sum'],'AMT_CREDIT_SUM_OVERDUE':['sum'],
             'DAYS_CREDIT_UPDATE':['max'],'AMT_CREDIT_MAX_OVERDUE':['sum'],'AMT_ANNUITY_active':['sum'],'bureauMOB':['min','max']})
        f.columns = [str("_").join(col).strip() for col in f.columns.values]
        a=0

if all:#joining with main train dataset
    data = pd.read_csv(sLoc + 'train.csv')
    #data = pd.read_csv(sLoc + 'application_test.csv')
    data=data.set_index('SK_ID_CURR').join(other=d,rsuffix="prev").reset_index()
    data = data.set_index('SK_ID_CURR').join(other=f,rsuffix="bur").reset_index()


#creating soem extra variable








    data.to_csv(dLoc+'final.csv')
    #data.to_csv('/home/pooja/PycharmProjects/datanalysis/finalDatasets/finalTest.csv')
    temp = distReports(data)
    report = pd.read_csv(dLocT + 'allDescribe.csv')
    temp['dataset'] = 'final.csv'
    report.append(temp).to_csv(dLocT + 'allDescribe.csv')








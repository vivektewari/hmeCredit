import pandas as pd
dLoc="/home/pooja/PycharmProjects/datanalysis/derivedData/"
task=2
if task==1:

    data=pd.read_csv('/home/pooja/PycharmProjects/datanalysis/relevantDatasets/installments_payments.csv')
    rollup=data.groupby(['SK_ID_PREV','NUM_INSTALMENT_NUMBER'])['AMT_PAYMENT'].sum().reset_index()
    rollup=data.drop_duplicates(['SK_ID_PREV','NUM_INSTALMENT_NUMBER']).drop(['AMT_PAYMENT'],axis=1).set_index(['SK_ID_PREV','NUM_INSTALMENT_NUMBER']).join(rollup.set_index(['SK_ID_PREV','NUM_INSTALMENT_NUMBER'])).reset_index()
    rollup.sort_values(inplace=True,by=['SK_ID_PREV','NUM_INSTALMENT_NUMBER'])
    rollup['underpaid']=rollup['AMT_INSTALMENT']-rollup['AMT_PAYMENT']
    rollup['percUnderpaid']=rollup['underpaid']/rollup['AMT_INSTALMENT']
    rollup['unpaid_trigger'] =  rollup['underpaid'].apply(lambda x:int(x>10))
    rollup['prepaid_trigger'] =  rollup['underpaid'].apply(lambda x:int(x<-10))
    rollup.to_csv(dLoc+'installments_payments.csv')
elif task==2 :
     data=pd.read_csv('/home/pooja/PycharmProjects/datanalysis/relevantDatasets/POS_CASH_balance.csv' )
    
     data.to_csv(dLoc+"trial.csv")
     c=0



elif task==3 :
    pass
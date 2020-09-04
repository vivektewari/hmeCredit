
import pandas as pd
holdMode=True
# s_application_train=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/rawDatas//application_train.csv",nrows=1000)
# s_bureau=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/rawDatas//bureau.csv")
# s_bureau_balance=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/rawDatas//bureau_balance.csv")
# s_credit_card_balance=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/rawDatas//credit_card_balance.csv")
# s_installments_payments=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/rawDatas//installments_payments.csv")
# s_POS_CASH_balance=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/rawDatas//POS_CASH_balance.csv")
# s_previous_application=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/rawDatas//previous_application.csv")
# #s_application_test=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/rawDatas//application_test.csv")
# subset=s_application_train['SK_ID_CURR'].to_list()
#print("total Train="+str(len(subset1)))
def groupFinder(subset1):
    train2install=set(subset1).intersection(set(s_installments_payments['SK_ID_CURR'].tolist()))
    train2bureau=set(subset1).intersection(set(s_bureau['SK_ID_CURR'].tolist()))
    train2Card=set(subset1).intersection(set(s_credit_card_balance['SK_ID_CURR'].tolist()))
    train2Pos=set(subset1).intersection(set(s_POS_CASH_balance['SK_ID_CURR'].tolist()))
    bureauonlyPos=train2bureau.intersection(train2Pos.difference(train2Card))
    bureauCard=train2bureau.intersection(train2Card)
    CardPos=train2Pos.intersection(train2Card)
    CardPosBur=CardPos.intersection(train2bureau)
    train2prev=set(subset1).intersection(set(s_previous_application['SK_ID_CURR'].tolist()))
    prev2Pos=train2prev.intersection(train2Pos)
    prev2Card=train2prev.intersection(train2Card)
    noCardPos=set(subset1).difference(train2Card.union(train2Pos))
    noHit=noCardPos.difference(train2bureau)

    statList=[train2install,train2bureau,train2Card,train2Pos,bureauonlyPos,bureauCard,CardPos,
              CardPosBur,train2prev,prev2Pos,prev2Card,noCardPos,noHit]
    return statList
# trainGroups=groupFinder(subset)
# #testGroups=groupFinder(subset2)
# for ele in trainGroups:
#     print(len(ele)/len(subset))
def num_tradelines(df,key,prod_key):
    return pd.DataFrame(df.groupby(key)[prod_key].count().reset_index().describe().transpose())
def past(df,key,months):
    a=len(df[key].unique())
    b=len(df[df[months]>-61][key].unique())
    return a,b
#s_previous_application=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets//previous_application.csv")
# b=num_tradelines(s_previous_application,'SK_ID_CURR','SK_ID_PREV')
# s_bureau=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets//bureau.csv")
# c=num_tradelines(s_bureau,'   SK_ID_CURR','SK_ID_BUREAU')
#s_POS_CASH_balance=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets//POS_CASH_balance.csv")
#d= s_POS_CASH_balance.groupby('MONTHS_BALANCE')['MONTHS_BALANCE'].count()
#q=past(s_POS_CASH_balance,'SK_ID_CURR','MONTHS_BALANCE')
s_bureau_balance=pd.read_csv("/home/pooja/PycharmProjects/datanalysis/relevantDatasets//bureau_balance.csv")
w=past(s_bureau_balance,'SK_ID_BUREAU','MONTHS_BALANCE')

d=0
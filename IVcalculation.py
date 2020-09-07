import warnings
from dataExploration import distReports,plotGrabh
from iv import iv_all,binning
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
loc="/home/pooja/PycharmProjects/datanalysis/finalDatasets/"
#

#train=pd.read_csv(loc+"relevantDatasets/"+"train.csv")
#loc="/home/pooja/PycharmProjects/datanalysis/bureau/"
train= pd.read_csv('/home/pooja/PycharmProjects/datanalysis/finalDatasets/final_withCross.csv')#pd.read_csv(loc+"bur.csv")
#a=plotGrabh(train,'TARGET',loc+"images/")
binned=binning(train,'TARGET',maxobjectFeatures=300)

ivData=iv_all(binned,'TARGET')

#writer = pd.ExcelWriter(loc+"iv3.xlsx")
#ivData.to_excel(writer,sheet_name="iv_detailed")
#ivData.groupby('variable')['ivValue'].sum().to_excel(writer,sheet_name="iv_summary")
ivData.to_csv(loc+"iv_detailed_cross.csv")
ivData.groupby('variable')['ivValue'].sum().to_csv(loc+"iv_sum_cross.csv")

# ivInfo=pd.read_csv(loc+"iv3.csv")
# distRepo=distReports(train,ivInfo)
# distRepo.to_csv(loc+"summary.csv")
#writer.save()
#writer.close()

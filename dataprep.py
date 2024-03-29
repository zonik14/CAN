
# IMPORTS
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score











#PRE-PROCESSING

#MERGE AND REMOVAL OF THE "TIME" AND "READING ID" VARIABLES

df = pd.read_csv("electrical_1.csv")
df2 = pd.read_csv("electrical_2.csv")
df3 = pd.read_csv("electrical_3.csv")
df4 = pd.read_csv("electrical_4.csv")
alldf = [df, df2, df3, df4]
eletrical_data = pd.concat(alldf, ignore_index=True, axis=0)
to_drop = ["Reading ID"]
eletrical_data = eletrical_data.drop(to_drop, axis=1)
eletrical_data["Cause"] = "electrical/clothing"
eletrical_data.to_csv("merged.csv")

eletrical_data

carton = pd.read_csv("carton_1.csv")
carton2 = pd.read_csv("carton_2.csv")
allcarton = [carton, carton2]
carton_data = pd.concat(allcarton, ignore_index=True, axis=0)
carton_data = carton_data.drop(to_drop, axis=1)
carton_data["Cause"] = "charcoal/cardboard"
carton_data.to_csv("mergedcarton.csv")

carton_data

clothing = pd.read_csv("clothing_1.csv")
clothing2 = pd.read_csv("clothing_2.csv")
allclothing = [clothing, clothing2]
clothing_data = pd.concat(allclothing, ignore_index=True, axis=0)
clothing_data = clothing_data.drop(to_drop, axis=1)
clothing_data["Cause"] = "charcoal/clothing"
clothing_data.to_csv("mergedclothing.csv")

clothing_data

allmerged = pd.concat([eletrical_data, carton_data, clothing_data], ignore_index= True)
allmerged.to_csv('allmerged.csv', index=False)


allmerged = allmerged.drop(['Status'], axis=1)

# Extract time features from the "Time Reading" column
time_readings = pd.to_datetime(allmerged['Time'], format='%H:%M:%S')
allmerged['Hour'] = time_readings.dt.hour
allmerged['Minute'] = time_readings.dt.minute
allmerged['Second'] = time_readings.dt.second
allmerged
#CHECKING FOR MISSING VALUES
print(allmerged.dtypes)
mv = allmerged.isnull().sum()
mv

#allmergedclean = allmerged.dropna()

#SPLITTING THE DATASET INTO FEATURES AND LABELS

x = allmerged.drop(['Time','Detector'], axis=1)
y = allmerged['Detector']

y= y.replace({'ON': 1, 'OFF': 0})
x['Cause'] = x['Cause'].astype('category')


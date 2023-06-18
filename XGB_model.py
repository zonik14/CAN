
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, average_precision_score

#Number of iterations
num_iterations = 100

#Initialize lists to store evaluation metrics
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
mAP_list = []

for _ in range(num_iterations):
    #Read the merged data
    allmerged = pd.read_csv('allmerged.csv')
    allmerged = allmerged.drop(['Status'], axis=1)
    time_readings = pd.to_datetime(allmerged['Time'], format='%H:%M:%S')
    allmerged['Hour'] = time_readings.dt.hour
    allmerged['Minute'] = time_readings.dt.minute
    allmerged['Second'] = time_readings.dt.second
   
    allmerged
    #Split the dataset into features and labels
    x = allmerged.drop(['Time', 'Detector'], axis=1)
    
    y = allmerged['Detector']
    y = y.replace({'ON': 1, 'OFF': 0})
    x['Cause'] = x['Cause'].astype('category')
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(tree_method="gpu_hist", enable_categorical=True)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    #EVALUATION METRICS
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    y_test_binary = y_test.astype(bool).astype(int)
    y_pred_binary = y_pred.astype(bool).astype(int)
    mAP = average_precision_score(y_test_binary, y_pred_binary, average='micro')

    
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    mAP_list.append(mAP)

#AVERAGE OF THE EVALUATION METRICS 
avg_accuracy = sum(accuracy_list) / num_iterations
avg_precision = sum(precision_list) / num_iterations
avg_recall = sum(recall_list) / num_iterations
avg_f1 = sum(f1_list) / num_iterations
avg_mAP = sum(mAP_list) / num_iterations

print("Average Accuracy:", avg_accuracy)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)
print("Average F1-score:", avg_f1)
print("Average mAP Score:", avg_mAP)

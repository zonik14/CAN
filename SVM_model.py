import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder

# Number of iterations
num_iterations = 100

# Initialize lists to store evaluation metrics
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
mAP_list = []

# Set the working directory
os.chdir("C:/Users/zemar/OneDrive/Ambiente de Trabalho/train")

# Load and preprocess the data
df = pd.read_csv("electrical_1.csv")
df2 = pd.read_csv("electrical_2.csv")
df3 = pd.read_csv("electrical_3.csv")
df4 = pd.read_csv("electrical_4.csv")
alldf = [df, df2, df3, df4]
eletrical_data = pd.concat(alldf, ignore_index=True, axis=0)
to_drop = ["Time", "Reading ID"]
eletrical_data = eletrical_data.drop(to_drop, axis=1)
eletrical_data["Cause"] = "electrical/clothing"
eletrical_data.to_csv("merged.csv")

carton = pd.read_csv("carton_1.csv")
carton2 = pd.read_csv("carton_2.csv")
allcarton = [carton, carton2]
carton_data = pd.concat(allcarton, ignore_index=True, axis=0)
carton_data = carton_data.drop(to_drop, axis=1)
carton_data["Cause"] = "charcoal/cardboard"
carton_data.to_csv("mergedcarton.csv")

clothing = pd.read_csv("clothing_1.csv")
clothing2 = pd.read_csv("clothing_2.csv")
allclothing = [clothing, clothing2]
clothing_data = pd.concat(allclothing, ignore_index=True, axis=0)
clothing_data = clothing_data.drop(to_drop, axis=1)
clothing_data["Cause"] = "charcoal/clothing"
clothing_data.to_csv("mergedclothing.csv")

allmerged = pd.concat([eletrical_data, carton_data, clothing_data], ignore_index=True)
allmerged.to_csv('allmerged.csv', index=False)

allmerged = allmerged.drop(['Status'], axis=1)

time_readings = pd.to_datetime(allmerged['Time'], format='%H:%M:%S')
allmerged['Hour'] = time_readings.dt.hour
allmerged['Minute'] = time_readings.dt.minute
allmerged['Second'] = time_readings.dt.second
allmerged
# Check for missing values
mv = allmerged.isnull().sum()

# Split the dataset into features and labels
x = allmerged.drop(['Time','Detector'], axis=1)
y = allmerged['Detector']

# Encode categorical variable 'Cause' using label encoding
label_encoder = LabelEncoder()
x['Cause'] = label_encoder.fit_transform(x['Cause'])

for _ in range(num_iterations):
    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize an SVM classifier
    model = SVC()

    # Fit the model to the training data
    model.fit(x_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(x_test)

    # Calculate the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    y_test_binary = y_test.astype(bool).astype(int)
    y_pred_binary = y_pred.astype(bool).astype(int)
    mAP = average_precision_score(y_test_binary, y_pred_binary, average='micro')

    # Append the metrics to the respective lists
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    mAP_list.append(mAP)

# Calculate the average of the evaluation metrics
avg_accuracy = sum(accuracy_list) / num_iterations
avg_precision = sum(precision_list) / num_iterations
avg_recall = sum(recall_list) / num_iterations
avg_f1 = sum(f1_list) / num_iterations
avg_mAP = sum(mAP_list) / num_iterations

# Print the average evaluation metrics
print("Average Accuracy:", avg_accuracy)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)
print("Average F1-score:", avg_f1)
print("Average mAP Score:", avg_mAP)

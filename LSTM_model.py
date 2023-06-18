import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, average_precision_score
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Set the working directory
os.chdir("C:/Users/zemar/OneDrive/Ambiente de Trabalho/train")

# Load and preprocess the data
df = pd.read_csv("electrical_1.csv")
df2 = pd.read_csv("electrical_2.csv")
df3 = pd.read_csv("electrical_3.csv")
df4 = pd.read_csv("electrical_4.csv")
alldf = [df, df2, df3, df4]
electrical_data = pd.concat(alldf, ignore_index=True, axis=0)
to_drop = ["Time", "Reading ID"]
electrical_data = electrical_data.drop(to_drop, axis=1)
electrical_data["Cause"] = "electrical/clothing"
electrical_data.to_csv("merged.csv")

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

allmerged = pd.concat([electrical_data, carton_data, clothing_data], ignore_index=True)
allmerged.to_csv('allmerged.csv', index=False)

allmerged = allmerged.drop(['Status'], axis=1)

time_readings = pd.to_datetime(allmerged['Time'], format='%H:%M:%S')
allmerged['Hour'] = time_readings.dt.hour
allmerged['Minute'] = time_readings.dt.minute
allmerged['Second'] = time_readings.dt.second

# Check for missing values
mv = allmerged.isnull().sum()

# Split the dataset into features and labels
x = allmerged.drop(['Time','Detector'], axis=1)
y = allmerged['Detector']
y = y.replace({'ON': 1, 'OFF': 0})

# Encode categorical variable 'Cause' using label encoding
label_encoder = LabelEncoder()
x['Cause'] = label_encoder.fit_transform(x['Cause'])

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train.astype(int)
x_test = x_test.astype(int)
 
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print(x_test.dtypes)
print(y_test.dtypes)

# Convert features and labels to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Normalize the features (optional but recommended for neural networks)
x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)

# Reshape the input data for LSTM (input shape: [samples, time steps, features])
time_steps = 1  # Number of time steps (can be adjusted as needed)
features = x_train.shape[1]
x_train = x_train.reshape(x_train.shape[0], time_steps, features)
x_test = x_test.reshape(x_test.shape[0], time_steps, features)

# Initialize variables for evaluation metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
classification_reports = []
mAP_scores = []

# Run the model 100 times
for i in range(100):
    # Initialize a sequential model
    model = Sequential()

    # Add LSTM layer
    model.add(LSTM(64, activation='relu', input_shape=(time_steps, features)))

    # Add output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fit the model to the training data
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Make predictions on the test data
    y_pred = model.predict(x_test)
    y_pred_classes = np.round(y_pred)

    # Calculate the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    accuracy_scores.append(accuracy)

    precision = precision_score(y_test, y_pred_classes)
    precision_scores.append(precision)

    recall = recall_score(y_test, y_pred_classes)
    recall_scores.append(recall)

    f1 = f1_score(y_test, y_pred_classes)
    f1_scores.append(f1)

    clf_report = classification_report(y_test, y_pred_classes)
    classification_reports.append(clf_report)

    mAP = average_precision_score(y_test, y_pred)
    mAP_scores.append(mAP)

# Calculate the average of evaluation metrics
avg_accuracy = np.mean(accuracy_scores)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)
avg_mAP = np.mean(mAP_scores)

# Print the average evaluation metrics
print("Average Accuracy:", avg_accuracy)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)
print("Average F1-score:", avg_f1)
print("Average mAP Score:", avg_mAP)

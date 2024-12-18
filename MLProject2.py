# 12/16/24
# Machine learning project using Microsoft optical backbone performance datasets
# and using it to train a One-Class SVM model to display anomalies (-1 value points)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# Read in the space-seperated data into pandas
df = pd.read_csv('MicrosoftWideAreaOpticalBackboneDataset/channel_1439_segment_37.txt', delim_whitespace=True, header=None)

# Add column names
df.columns = ['Timestamp', 'Q_Factor', 'Transmit_Power', 'Chromatic_Dispersion', 'Polarization_Mode_Dispersion']

# Convert the timestamp column to datetime type
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y.%m.%d.%H.%M.%S')

# Standardize the feature columns
features = df[['Q_Factor', 'Transmit_Power', 'Chromatic_Dispersion', 'Polarization_Mode_Dispersion']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Apply One-Class SVM for anomaly detection
oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1) #nu is the fraction of outliers
df['One_Class_SVM_Outliers'] = oc_svm.fit_predict(scaled_data)

# -1 = anomaly /// 1 = normal point

# filter so we only display results
outliers = df[df['One_Class_SVM_Outliers'] == -1]


# Display the results
print(outliers[['Timestamp', 'Q_Factor', 'Transmit_Power', 'Chromatic_Dispersion', 'Polarization_Mode_Dispersion', 'One_Class_SVM_Outliers']])


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load and prepare your dataset
dataset = pd.read_csv(r'indiancrop_dataset.csv')
x = dataset.drop('CROP', axis=1).values
y = dataset['CROP']

# Data preprocessing
ms = MinMaxScaler()
x = ms.fit_transform(x)

# Model training
rfc = RandomForestClassifier()
rfc.fit(x, y)

# Save the model and scaler to pickle files
with open('model.pkl', 'wb') as model_file:
    pickle.dump(rfc, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(ms, scaler_file)

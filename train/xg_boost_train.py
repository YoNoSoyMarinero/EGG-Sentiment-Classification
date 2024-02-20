import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Validation shape: {val_df.shape}")

train_df = train_df.drop("Unnamed: 0", axis=1)
val_df = val_df.drop("Unnamed: 0", axis=1)
test_df = test_df.drop("Unnamed: 0", axis=1)

train_df = train_df[train_df['label'] != 0]
val_df = val_df[val_df['label'] != 0]
test_df = test_df[test_df['label'] != 0]

print('Train labels: ', train_df['label'].unique())
print('Validation labels: ', val_df['label'].unique())
print('Test labels: ', test_df['label'].unique())

print(f"Train shape after dropping 0: {train_df.shape}")
print(f"Test shape after dropping 0: {test_df.shape}")
print(f"Validation shape after dropping 0: {val_df.shape}")

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']
X_val = val_df.drop('label', axis=1)
y_val = val_df['label']

X_train = pd.concat([X_train, X_val], axis=0)
y_train = pd.concat([y_train, y_val], axis=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
dtest = xgb.DMatrix(X_test, label=y_test_encoded)

param = {
    'objective': 'multi:softmax',  
    'num_class': 5,                 
    'max_depth': 10,                 
    'min_child_weight': 10,
    'eta': 0.01,                    
    'eval_metric': 'merror',         
    'subsample': 0.5,
    'tree_method': 'approx'
}

num_round = 1000
model = xgb.train(param, dtrain, num_round)

y_pred = model.predict(dtest)

y_pred_original = label_encoder.inverse_transform(y_pred.astype(int))

accuracy = accuracy_score(y_test, y_pred_original)
print(f'Accuracy: {accuracy:.2f}')

print(classification_report(y_test, y_pred_original))
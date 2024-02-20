import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle


train_df = pd.read_csv('data/dataset_2000/train.csv')
val_df = pd.read_csv('data/dataset_2000/val.csv')
test_df = pd.read_csv('data/dataset_2000/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Validation shape: {val_df.shape}")


train_df = train_df.drop("Unnamed: 0", axis=1)
val_df = val_df.drop("Unnamed: 0", axis=1)
test_df = test_df.drop("Unnamed: 0", axis=1)

train_df.drop(train_df[train_df['label'] == 0].index, inplace=True)
val_df.drop(val_df[val_df['label'] == 0].index, inplace=True)
test_df.drop(test_df[test_df['label'] == 0].index, inplace=True)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Validation shape: {val_df.shape}")

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']
X_val = val_df.drop('label', axis=1)
y_val = val_df['label']


model = SVC(C=1, kernel='rbf')
model.fit(X_train, y_train)
with open('models/svm_model_1_rbf.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Model Accuracy: {accuracy}")

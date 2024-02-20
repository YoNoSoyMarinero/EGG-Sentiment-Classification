import tensorflow as tf
from keras import layers, Sequential
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Define the model
def create_model(input_shape, num_classes):
    model = Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Multiclass classification, so using softmax activation
    ])
    return model

# Load data (replace this with your EEG data loading code)
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')

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

X_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values
X_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].values
X_val = val_df.drop('label', axis=1).values
y_val = val_df['label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Determine the number of classes
num_classes = len(set(y_train))

# Define model and compile it
input_shape = X_train.shape[1:]  # Shape of input data (excluding batch dimension)
model = create_model(input_shape, num_classes+1)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Predictions on test set
y_pred = model.predict_classes(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {test_accuracy}')

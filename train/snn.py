import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

class TabularNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(TabularNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.bn1(torch.tanh(self.fc1(x)))
        x = self.dropout1(x)
        x = self.bn2(torch.tanh(self.fc2(x)))
        x = self.dropout2(x)
        x = self.bn3(torch.tanh(self.fc3(x)))
        x = self.dropout3(x)
        x = self.softmax(self.fc4(x))
        return x


train_df = pd.read_csv('../data/train.csv')
val_df = pd.read_csv('../data/val.csv')
test_df = pd.read_csv('../data/test.csv')

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
y_train = train_df['label'].astype(int).values
X_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].astype(int).values
X_val = val_df.drop('label', axis=1).values
y_val = val_df['label'].astype(int).values

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train - 1)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val - 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test - 1)

# Create PyTorch datasets and loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32
combined_train_val_dataset = ConcatDataset([train_dataset, val_dataset])
train_val_loader = DataLoader(combined_train_val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model
input_size = X_train.shape[1]
hidden_size1 = 128
hidden_size2 = 128
hidden_size3 = 128
output_size = 5
model = TabularNN(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop on both train and validation data
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_val_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
#done
# Test loop
model.eval()
all_preds_test = []
all_labels_test = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predictions_test = torch.max(outputs, 1)
        all_preds_test.extend(predictions_test.cpu().numpy())
        all_labels_test.extend(labels.cpu().numpy())
       
accuracy_test = accuracy_score(all_labels_test, all_preds_test)
print(f"Test Accuracy: {accuracy_test}")

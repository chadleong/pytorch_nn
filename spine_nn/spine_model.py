import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

df = pd.read_csv("Dataset_spine.csv")
# print(df.head())
sns.countplot(x="Class_att", data=df)

# remap labels
df["Class_att"] = df["Class_att"].astype("category")
encode_map = {"Abnormal": 1, "Normal": 0}

df["Class_att"].replace(encode_map, inplace=True)

# print(df.head())

X = df.iloc[:, 0:-2]
y = df.iloc[:, -2]

# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001

## train data
class trainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


train_data = trainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
## test data
class testData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


test_data = testData(torch.FloatTensor(X_test))


train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)


class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(12, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# model = binaryClassification()
# model.to(device)
# print(model)
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# def binary_acc(y_pred, y_test):
#     y_pred_tag = torch.round(torch.sigmoid(y_pred))

#     correct_results_sum = (y_pred_tag == y_test).sum().float()
#     acc = correct_results_sum / y_test.shape[0]
#     acc = torch.round(acc * 100)

#     return acc


# model.train()
# for e in range(1, EPOCHS + 1):
#     epoch_loss = 0
#     epoch_acc = 0
#     for X_batch, y_batch in train_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         optimizer.zero_grad()

#         y_pred = model(X_batch)

#         loss = criterion(y_pred, y_batch.unsqueeze(1))
#         acc = binary_acc(y_pred, y_batch.unsqueeze(1))

#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()
#         epoch_acc += acc.item()

#     print(f"Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}")


# torch.save(model.state_dict(), "spine_model.pth")
# print("Saved PyTorch Model State to model.pth")

model = binaryClassification()
model.load_state_dict(torch.load("spine_model.pth"))
model.to(device)
y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

print(y_pred_list)

cm = confusion_matrix(y_test, y_pred_list, labels=[1.0, 0])

print(cm)

disp = ConfusionMatrixDisplay(cm, display_labels=[1, 0])

disp.plot()
plt.show()
print(classification_report(y_test, y_pred_list))

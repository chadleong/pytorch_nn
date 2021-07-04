import numpy as np
import pandas as pd
import seaborn as sns
from seaborn.external.husl import xyz_to_luv
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

df = pd.read_csv("winequality-red.csv")
print(df.head())

sns.countplot(x="quality", data=df)

# plt.show()

X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]


# Train - Test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)
# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21
)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)


def get_class_distribution(obj):
    count_dict = {
        "rating_3": 0,
        "rating_4": 0,
        "rating_5": 0,
        "rating_6": 0,
        "rating_7": 0,
        "rating_8": 0,
    }

    for i in obj:
        if i == 3:
            count_dict["rating_3"] += 1
        elif i == 4:
            count_dict["rating_4"] += 1
        elif i == 5:
            count_dict["rating_5"] += 1
        elif i == 6:
            count_dict["rating_6"] += 1
        elif i == 7:
            count_dict["rating_7"] += 1
        elif i == 8:
            count_dict["rating_8"] += 1
        else:
            print("Check classes.")

    return count_dict


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 7))
# Train
sns.barplot(
    data=pd.DataFrame.from_dict([get_class_distribution(y_train)]).melt(),
    x="variable",
    y="value",
    hue="variable",
    ax=axes[0],
).set_title("Class Distribution in Train Set")
# Val
sns.barplot(
    data=pd.DataFrame.from_dict([get_class_distribution(y_val)]).melt(),
    x="variable",
    y="value",
    hue="variable",
    ax=axes[1],
).set_title("Class Distribution in Val Set")
# Test
sns.barplot(
    data=pd.DataFrame.from_dict([get_class_distribution(y_test)]).melt(),
    x="variable",
    y="value",
    hue="variable",
    ax=axes[2],
).set_title("Class Distribution in Test Set")


# plt.show()
y_train, y_test, y_val = y_train.astype(float), y_test.astype(float), y_val.astype(float)


class RegressionDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


train_dataset = RegressionDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
val_dataset = RegressionDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
test_dataset = RegressionDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

EPOCHS = 400
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_FEATURES = len(X.columns)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)


class MultipleRegression(nn.Module):
    def __init__(self, num_features):
        super(MultipleRegression, self).__init__()

        self.layer_1 = nn.Linear(num_features, 16)
        self.layer_2 = nn.Linear(16, 32)
        self.layer_3 = nn.Linear(32, 16)
        self.layer_out = nn.Linear(16, 1)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_out(x)
        return x

    def predict(self, test_inputs):
        x = self.relu(self.layer_1(test_inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_out(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = MultipleRegression(NUM_FEATURES)
model.to(device)
print(model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_stats = {"train": [], "val": []}

print("Begin training.")
for e in tqdm(range(1, EPOCHS + 1)):
    # TRAINING
    train_epoch_loss = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()

        y_train_pred = model(X_train_batch)

        train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))

        train_loss.backward()
        optimizer.step()

        train_epoch_loss += train_loss.item()

    # VALIDATION
    with torch.no_grad():

        val_epoch_loss = 0

        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

            y_val_pred = model(X_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))

            val_epoch_loss += val_loss.item()
    loss_stats["train"].append(train_epoch_loss / len(train_loader))
    loss_stats["val"].append(val_epoch_loss / len(val_loader))

    print(
        f"Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}"
    )


train_val_loss_df = (
    pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=["index"]).rename(columns={"index": "epochs"})
)
plt.figure(figsize=(15, 8))
sns.lineplot(data=train_val_loss_df, x="epochs", y="value", hue="variable").set_title("Train-Val Loss/Epoch")
plt.show()

torch.save(model.state_dict(), "wine_model.pth")
print("Saved PyTorch Model State to model.pth")
# model.load_state_dict(torch.load("wine_model.pth"))
# model.to(device)


# y_pred_list = []
# with torch.no_grad():
#     model.eval()
#     for X_batch, _ in test_loader:
#         # print(X_batch)
#         X_batch = X_batch.to(device)
#         y_test_pred = model(X_batch)

#         print(y_test_pred)
# y_pred_list.append(y_test_pred.cpu().numpy())
# y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
# cm = confusion_matrix(y_test, y_pred_list)

# print(cm)

# print(y_pred_list)
# model.to(device)

# ans = [10.5, 0.24, 0.47, 2.1, 0.066, 6, 24, 0.9978, 3.15, 0.9, 11]
ans = [7.3, 0.98, 0.05, 2.1, 0.061, 20, 49, 0.99705, 3.31, 0.55, 9.7]

ans_s = scaler.transform([ans])
print(model(torch.FloatTensor(ans_s).to(device)))


# mse = mean_squared_error(y_test, y_pred_list)
# r_square = r2_score(y_test, y_pred_list)
# print("Mean Squared Error :", mse)
# print("R^2 :", r_square)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("tkagg")
import seaborn as sns
import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Reading data
df = pd.read_csv("barrier.csv")

# Encode Output Class
class2idx = {"Primary": 0, "Secondary": 1, "Common": 2}
idx2class = {v: k for k, v in class2idx.items()}
df["barrier"].replace(class2idx, inplace=True)


# Create Input and Output Data
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
# Train - Test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)
# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21
)

# Normalize Input
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)


def get_class_distribution(obj):
    count_dict = {
        "primary": 0,
        "secondary": 0,
        "common": 0,
    }

    for i in obj:
        if i == 0:
            count_dict["primary"] += 1
        elif i == 1:
            count_dict["secondary"] += 1
        elif i == 2:
            count_dict["common"] += 1

        else:
            print("Check classes.")

    return count_dict


#### Class distribution
def show_class_dis():
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

    plt.show()


class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

# Weighted sampling
target_list = []
for _, t in train_dataset:
    target_list.append(t)

target_list = torch.tensor(target_list)
target_list = target_list[torch.randperm(len(target_list))]
class_count = [i for i in get_class_distribution(y_train).values()]
class_weights = 1.0 / torch.tensor(class_count, dtype=torch.float)
# print(class_weights)


class_weights_all = class_weights[target_list]
weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all, num_samples=len(class_weights_all), replacement=True
)

# Model parameters
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.0007
NUM_FEATURES = len(X.columns)
NUM_CLASSES = 3  # primary, secondary, common

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)

# Define Neural Net Architecture
class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MulticlassClassification(num_feature=NUM_FEATURES, num_class=NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# print(model)

# Calculating accuracy
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


# Storing accuracy/epoch , loss/epoch
accuracy_stats = {"train": [], "val": []}
loss_stats = {"train": [], "val": []}


##Training model
def train():
    print("Begin training.")
    for e in tqdm.tqdm(range(1, EPOCHS + 1)):

        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()  # tells pytorch in training mode
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
        loss_stats["train"].append(train_epoch_loss / len(train_loader))
        loss_stats["val"].append(val_epoch_loss / len(val_loader))
        accuracy_stats["train"].append(train_epoch_acc / len(train_loader))
        accuracy_stats["val"].append(val_epoch_acc / len(val_loader))

        print(
            f"Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}"
        )


### plotting loss and acccuracy
def loss_acc():
    train_val_acc_df = (
        pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=["index"]).rename(columns={"index": "epochs"})
    )
    train_val_loss_df = (
        pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=["index"]).rename(columns={"index": "epochs"})
    )
    # Plot the dataframes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
    sns.lineplot(data=train_val_acc_df, x="epochs", y="value", hue="variable", ax=axes[0]).set_title(
        "Train-Val Accuracy/Epoch"
    )
    sns.lineplot(data=train_val_loss_df, x="epochs", y="value", hue="variable", ax=axes[1]).set_title(
        "Train-Val Loss/Epoch"
    )
    plt.show()


##Validation
def validate_model():
    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)

            y_test_pred = model(X_batch)
            # print(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim=1)

            y_pred_list.append(y_pred_tags.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list)).rename(columns=idx2class, index=idx2class)

    sns.heatmap(confusion_matrix_df, annot=True)

    print(classification_report(y_test, y_pred_list))
    plt.show()


### Show info
# print(df.head())
# sns.countplot(x="barrier", data=df)
# plt.show()

#### Show class dist
# show_class_dis()

# #### Train model
# train()

# # ### Show accuracy
# loss_acc()

# # ### Validate model
# validate_model()

# # ###save model
# torch.save(model.state_dict(), "barrier_model_multiclass.pth")
# print("Saved PyTorch Model State to model.pth")


####Load model
model.load_state_dict(torch.load("barrier_model_multiclass.pth"))
model.to(device)


##Prediction
ans = [0, 1800, 133.2, 6.6, 40, 13, 0.9978, 3.51, 0.56, 0.075, 9.4]
ans_s = scaler.transform([ans])
# print(torch.FloatTensor(ans_s))

model.eval()
# print(model(torch.FloatTensor(ans_s).to(device)))
res = torch.max(model(torch.FloatTensor(ans_s).to(device)), dim=1)

res = res[1].cpu().numpy()[0]

print("Barrier: " + str(idx2class[res]))


print("Done")

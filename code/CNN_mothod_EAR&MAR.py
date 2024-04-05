import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sklearn
from sklearn.model_selection import train_test_split, cross_val_score

#code for gpu training on COLAB
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ture_Set = np.load("dataset/Drowsy/EAR&MAR_D.npy")
false_Set = np.load("dataset/Non_drowsy/EAR&MAR_ND.npy")
#combine the ture&false set to a tensor with size [40000, 4]
dataset = np.vstack((ture_Set[:, :3], false_Set[:, :3]))
labelset = np.hstack((ture_Set[:, 3], false_Set[:, 3]))

#relabel targets
labelset[labelset == 'Drowsy'] = 1
labelset[labelset == 'Non_drowsy'] = 0
dataset = dataset.astype(np.float32)
labelset = labelset.astype(np.int64)

dataset = torch.tensor(dataset)
dataset = dataset.unsqueeze(2)
labelset = torch.tensor(labelset)

#divide train data to train and test
X_train, X_test, Y_train, Y_test = train_test_split(dataset, labelset, test_size=0.3, random_state=42)
# transport X_train, Y_train to GPU
X_train, Y_train = X_train.to(device), Y_train.to(device)
X_test, Y_test = X_test.to(device), Y_test.to(device)

# initialize the dataloader
train_data = TensorDataset(X_train, Y_train)
test_data = TensorDataset(X_test, Y_test)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=1 , padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 1, 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        output = self.softmax(x)

        return output

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.to(device)

def train():
    epochs = 10
    total_loss = 0.0
    total_samples = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_loss += loss.item() * len(inputs)
            total_samples += len(inputs)
        average_loss = (total_loss / total_samples) * 100
        print(f"Epoch {epoch + 1}, Running Loss: {running_loss}, Average Loss: {average_loss}%")
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到 GPU 上
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%")


#torch.save(model, "CNN_mothod_EAR&MAR.pth")


def main():
    train()


if __name__ == "__main__":
    main()







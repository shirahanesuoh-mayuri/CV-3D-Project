import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sklearn
from sklearn.model_selection import train_test_split, cross_val_score


ture_Set = np.load("dataset/Drowsy/EAR&MAR.npy")
false_Set = np.load("dataset/Non_drowsy/EAR&MAR.npy")
#combine the ture&false set to a tensor with size [40000, 4]
dataset = np.vstack((ture_Set[:, :3], false_Set[:, :3]))
labelset = np.hstack((ture_Set[:, 3], false_Set[:, 3]))

#relabel targets
labelset[labelset == 'Drowsy'] = 1
labelset[labelset == 'Non_drowsy'] = -1
dataset = dataset.astype(np.float64)
labelset = labelset.astype(int)

dataset = torch.tensor(dataset)
labelset = torch.tensor(labelset)

#divide train data to train and test
X_train, X_test, Y_train, Y_test = train_test_split(dataset, labelset, test_size=0.3, random_state=42)

# initialize the dataloader
train_data = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)








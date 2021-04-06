import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from TestModel import TestModel
from dataset import Dataset
import helper as H


dataset = Dataset()
X, y = dataset.preprocess()
X_train,  X_test,  y_train,  y_test = dataset.split( X,  y, fraction=0.3)
X_train,  X_test = dataset.scale(X_train, X_test)
train_loader,  test_loader = H.load( X_train,  X_test, y_train,  y_test)


testmodel = TestModel(input_dims=[12], output_dims=2, lr=0.01,
                                num_layers=9, num_nodes=6, trainloader = train_loader, testloader=test_loader)

acc, loss = testmodel.train()
print(acc, loss)
acc, loss = testmodel.test()
print(acc, loss)
print("===========================================")
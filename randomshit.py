#
# random shit goes here
#
import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Testmodel import TestModel
from dataset import Dataset
import helper as H
testmodel = TestModel(input_dims=[2], output_dims=2, lr=0.01,
                                num_layers=4, num_nodes=6)

dataset = Dataset()
X, y = dataset.preprocess()
X_train,  X_test,  y_train,  y_test =  dataset.split( X,  y, fraction=0.2)
X_train,  X_test = dataset.scale( X_train,  X_test)
train_loader,  test_loader = H.load( X_train,  X_test, y_train,  y_test)

acc, loss = testmodel.train(train_loader)
print(acc, loss)
acc, loss = testmodel.test(test_loader)
print(acc, loss)

'''
dataset = Dataset()
X, y = dataset.preprocess()
X_train,  X_test,  y_train,  y_test =  dataset.split( X,  y, fraction=0.2)
X_train,  X_test = dataset.scale( X_train,  X_test)
train_loader,  test_loader = H.load( X_train,  X_test, y_train,  y_test)

input_dims = [X.shape[1]]
output_dims = len(np.unique(y))
testmodel = TestModel.TestModel(input_dims=input_dims, output_dims=output_dims, 
                                lr=0.01, num_layers=4, num_nodes=6)

acc, loss = testmodel.train(train_loader)
print(acc, loss)
acc, loss = testmodel.test(test_loader)
print(acc, loss)

t = T.ones((2))
# print(testmodel.forward(t))
# print(testmodel.fcs[0].weight, testmodel.fcs[0].weight.shape)
print(testmodel.fcs)
print(testmodel.output)

testmodel.add_layers(2)
testmodel.remove_neurons(3)

print(testmodel.fcs[0].weight, testmodel.fcs[0].weight.shape)
print(testmodel.fcs[1].weight, testmodel.fcs[1].weight.shape)
print(testmodel.fcs[2].weight, testmodel.fcs[2].weight.shape)
print(testmodel.output.weight, testmodel.output.weight.shape)

print(testmodel(t))
print(testmodel.fcs)
print(testmodel.output)
# print(testmodel.forward(t))

t = T.ones((2))


#print(testmodel.forward(t))
# print(testmodel.fcs[0].weight, testmodel.fcs[0].weight.shape)
print(testmodel.fcs)
print(testmodel.output)

testmodel.add_layers(2)
testmodel.remove_neurons(3)

print(testmodel.fcs[0].weight, testmodel.fcs[0].weight.shape)
print(testmodel.fcs[1].weight, testmodel.fcs[1].weight.shape)
print(testmodel.fcs[2].weight, testmodel.fcs[2].weight.shape)
print(testmodel.output.weight, testmodel.output.weight.shape)


print(testmodel(t))
print(testmodel.fcs)
print(testmodel.output)
#print(testmodel.forward(t))
'''
#
# random stuff goes here
#
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import TestModel


testmodel = TestModel.TestModel(input_dims=[2], output_dims=2, lr=0.01,
                                num_layers=4, num_nodes=6)

t = T.ones((2))
#print(testmodel.forward(t))
# print(testmodel.fcs[0].weight, testmodel.fcs[0].weight.shape)
print(testmodel.fcs)
print(testmodel.output)
'''
testmodel.add_layers(2)
testmodel.remove_neurons(3)

print(testmodel.fcs[0].weight, testmodel.fcs[0].weight.shape)
print(testmodel.fcs[1].weight, testmodel.fcs[1].weight.shape)
print(testmodel.fcs[2].weight, testmodel.fcs[2].weight.shape)
print(testmodel.output.weight, testmodel.output.weight.shape)
'''

print(testmodel(t))
print(testmodel.fcs)
print(testmodel.output)
#print(testmodel.forward(t))


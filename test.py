#
# random stuff goes here
#
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import TestModel
from dataset import Dataset
import helper as H
import constants as C
from agent import DDPGAgent
from multiAgent import MADDPG
from environment import Environment

'''
from environment import Environment
env = Environment()
#s, r = env.reset()
#print(s, r)

s_, r = env.step(-1, 0)
s1_, r1 = env.step(3, 1)
synch_state = env.synch()
print(synch_state)



agent1 = DDPGAgent(alpha=C.ALPHA, beta=C.BETA, tau=C.TAU, input_dims=[2] ,
                   n_actions=1, hd1_dims = 40, hd2_dims = 30, mem_size = 1000,
                   gamma = 0.99, batch_size = 16)
print(agent1)
#agent2 = DDPGAgent(alpha=C.ALPHA, beta=C.BETA, tau=C.TAU, input_dims=[2] ,
#                   n_actions=1, hd1_dims = 40, hd2_dims = 30, mem_size = 1000,
#                   gamma = 0.99, batch_size = 16)
#print(agent2)
print(agent1.choose_action((3, 16)))
agent1.store_transition([1,3], 3, 4, [4, 3])
agent1.store_transition([2,5], 5, 3, [7, 3])
agent1.learn()
'''

#env = Environment()

#call multiAgent here
#controller = MADDPG(env, num_agents=C.NUM_AGENTS, alpha=C.ALPHA, beta=C.BETA, tau=C.TAU, input_dims=[C.NUM_AGENTS] ,n_actions=C.N_ACTIONS, hd1_dims = #C.H1_DIMS, hd2_dims = C.H2_DIMS, mem_size = C.BUF_LEN,gamma = C.GAMMA, batch_size = C.BATCH_SIZE)

#controller.run()

#print("hello")

from environment import Environment
env = Environment()
#print(env.model1)
print("Initial model")
param = iter(env.model1.parameters())
for p in param:
    print(p)
#print()

print(env.model1.optimizer)
print("\n===================================\n\n")
print("Initial model + 2 layers")
env.step(2, 0)

param = iter(env.model1.parameters())
for p in param:
    print(p)
#print()

print(env.model1.optimizer)

print("\n===================================\n\n")
#env.step(0, 0)
env.model1.train()

param = iter(env.model1.parameters())
for p in param:
    print(p)
#print()

print(env.model1.optimizer)

print("===================================")

'''
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
'''
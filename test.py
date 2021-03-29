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

'''
from environment import Environment
env = Environment()
#s, r = env.reset()
#print(s, r)

s_, r = env.step(-1, 0)
s1_, r1 = env.step(3, 1)
synch_state = env.synch()
print(synch_state)
'''


agent1 = DDPGAgent(alpha=C.ALPHA, beta=C.BETA, tau=C.TAU, input_dims=[2] ,
                   n_actions=1, hd1_dims = 40, hd2_dims = 30, mem_size = 1000,
                   gamma = 0.99, batch_size = 16)
print(agent1)
agent2 = DDPGAgent(alpha=C.ALPHA, beta=C.BETA, tau=C.TAU, input_dims=[2] ,
                   n_actions=1, hd1_dims = 40, hd2_dims = 30, mem_size = 1000,
                   gamma = 0.99, batch_size = 16)
print(agent2)

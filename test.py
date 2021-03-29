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
from environment import Environment

env = Environment()

#call multiAgent here
controller = MADDPG(env, num_agents=C.NUM_AGENTS, alpha=C.ALPHA, beta=C.BETA, tau=C.TAU, input_dims=[C.NUM_AGENTS] ,n_actions=C.N_ACTIONS, hd1_dims = C.H1_DIMS, hd2_dims = C.H2_DIMS, mem_size = C.BUF_LEN,gamma = C.GAMMA, batch_size = C.BATCH_SIZE)

controller.run()


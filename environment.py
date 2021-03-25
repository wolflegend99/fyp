import gym
import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import dataset
import constants as C
print(dir)


class Env(gym.Env):
  def __init__(self, dataset):
    self.dataset = Dataset()
    self.X, self.y = preprocess(dataset)
    self.model1 = Model(params go here)
    self.model2 = Model(params go here)
    self.model3 = Model(params go here)
'''
  def reset(self):
    layers = T.rand(C.MIN_HIDDEN_LAYERS, C.MAX_HIDDEN_LAYERS)
    neurons = T.rand(C.MIN_NODES, C.MAX_NODES)
    lr = T.uniform(u, sigma)
    self.model1.initialise(layers, neurons, lr)
    self.model2.initialise(layers, neurons, lr)
    self.model3.initialise(layers, neurons, lr)
  
  def step(self, action, agent_no):
    if agent_no == 1:
      state_, reward = change_layers(action)
    elif agent_no == 2:
      state_, reward =  change_neurons(action)
    elif agent_no == 3:
      state_, reward = change_lr(action)
    
    if steps % C.SYNCH_STEP == 0
      self.synch()
    
    return state_, reward
  
  def synch(self):
    pass
  
  def change_layer(self, action):
    next_state = self.model1.change_layer()
    reward = self.model1.train()
    # reward = reward - punishment
    return next_state, reward
  
  def change_neuron(self, action):
    pass
  
  def change_lr(self, action):
    pass
  
  def seed(self):
    pass 
'''
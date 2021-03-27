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
import helper as H
from TestModel import TestModel

class Environment(gym.Env):
  def __init__(self, dataset):
    self.dataset = Dataset()
    self.X, self.y = self.dataset.preprocess()
    self.X_train, self.X_test, self.y_train, self.y_test = self.dataset.split(self.X, self.y, fraction=0.2)
    self.X_train, self.X_test = self.dataset.scale(self.X_train, self.X_test)
    self.train_loader, self.test_loader = H.load(self.X_train, self.X_test,
                                                 self.y_train, self.y_test)
    
    input_dims = self.X.shape[1]
    output_dims = len(np.unique(self.y))
    
    self.model1 = TestModel(input_dims, output_dims, 0.01, 3, 16)
    self.model2 = TestModel(input_dims, output_dims, 0.01, 3, 16)

  def reset(self):
    layers = np.random.randint(C.MIN_HIDDEN_LAYERS, C.MAX_HIDDEN_LAYERS)
    neurons = np.random.randint(C.MIN_NODES, C.MAX_NODES)
    lr = T.uniform(u, sigma)
    self.model1.initialise(layers, neurons, lr)
    self.model2.initialise(layers, neurons, lr)
    self.model3.initialise(layers, neurons, lr)

  def step(self, action, agent_no):
    if agent_no == 1:
      state_, reward = change_layers(action)
    elif agent_no == 2:
      state_, reward =  change_neurons(action)
    
    if steps % C.SYNCH_STEP == 0:
      self.synch()
    
    return state_, reward
  
  def synch(self):
    pass
  
  def change_layer(self, action):
    
    next_state = self.model1.change_layer()
    get_sample = sample(X_train, limit = C.SAMPLE_SIZE)
    train_acc = self.model1.train(self.train_loader)
    test_acc, over_estimate = self.model1.test(self.test_loader)
    reward = H.reward(test_acc, over_estimate, train_acc, params...)
    # reward = reward - punishment
    return next_state, reward
  
  def change_neuron(self, action):
    next_state = self.model2.change_neuron()
    get_sample = sample(X_train, limit = C.SAMPLE_SIZE)
    
    reward = self.model2.train(sample)
    # reward = reward - punishment
    return next_state, reward
  
  def change_lr(self, action):
    next_state = self.model3.change_lr()
    get_sample = sample(X_train, limit = C.SAMPLE_SIZE)
    
    reward = self.model3.train(sample)
    # reward = reward - punishment
    return next_state, reward
  
  def seed(self):
    pass 

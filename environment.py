import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from dataset import Dataset
import constants as C
import helper as H
from TestModel import TestModel

class Environment():
  def __init__(self, path='churn_modelling.csv', ):
    self.path = path
    self.dataset = Dataset(path = self.path)
    self.X, self.y = self.dataset.preprocess()
    self.X_train, self.X_test, self.y_train, self.y_test = self.dataset.split(self.X, self.y, fraction=0.2)
    self.X_train, self.X_test = self.dataset.scale(self.X_train, self.X_test)
    self.train_loader, self.test_loader = H.load(self.X_train, self.X_test,
                                                 self.y_train, self.y_test)
    
    input_dims = [self.X.shape[1]]
    output_dims = len(np.unique(self.y))
    
    self.model1 = TestModel(input_dims, output_dims, 0.01, 3, 16)
    self.model2 = TestModel(input_dims, output_dims, 0.01, 3, 16)

  def reset(self):
    layers = np.random.randint(C.MIN_HIDDEN_LAYERS, C.MAX_HIDDEN_LAYERS)
    neurons = np.random.randint(C.MIN_NODES, C.MAX_NODES)
    self.model1.initialise(layers, neurons)
    self.model2.initialise(layers, neurons)
    
    return [layers, neurons]

  def step(self, action, agent_no):
    if agent_no == 0:
      state_, reward = self.change_layers(action)
    elif agent_no == 1:
      state_, reward =  self.change_neurons(action)
    
    return state_, reward
  
  def synch(self):
    model1_layers = self.model1.num_layers
    model1_neurons = self.model1.num_nodes
    model2_layers = self.model2.num_layers
    model2_neurons = self.model2.num_nodes
    
    model1_action = model2_neurons - model1_neurons
    if(model1_action >= 0):
        self.model1.add_neurons(model1_action)
    else:
        self.model1.remove_neurons(-model1_action)
    
    model2_action = model1_layers - model2_layers
    if(model2_action >= 0):
        self.model2.add_layers(model2_action)
    else:
        self.model2.remove_layers(-model2_action)
    
    return [self.model2.num_layers, self.model1.num_nodes]
    
    
  
  def change_layers(self, action):
    
    if action >= 0:
        next_state = self.model1.add_layers(action)
    else:
        next_state = self.model1.remove_layers(-action)
    train_acc, train_loss = self.model1.train(self.train_loader)
    test_acc, test_loss = self.model1.test(self.test_loader)
    reward = H.reward(train_acc, train_loss,
                      test_acc, test_loss,
                      next_state[0])
    # reward = reward - punishment
    return next_state, reward
  
  def change_neurons(self, action):
    
    if action >= 0:
        next_state = self.model2.add_neurons(action)
    else:
        next_state = self.model2.remove_neurons(-action)
    train_acc, train_loss = self.model2.train(self.train_loader)
    test_acc, test_loss = self.model2.test(self.test_loader)
    reward = H.reward(train_acc, train_loss,
                      test_acc, test_loss, 
                      next_state[1])
    return next_state, reward
  
 
  def seed(self):
    pass 

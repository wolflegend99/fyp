import numpy as np
import pandas as pd
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statistics import mean
import math

class TestModel(nn.Module):
    def __init__(self, input_dims, output_dims, lr, num_layers, num_nodes):
        super(TestModel, self).__init__()
        self.input_dims = input_dims
        self.lr = lr
        self.num_layers = num_layers
        self.output_dims = output_dims
        self.num_nodes = num_nodes
        self.fcs = None
        self.output = None
        self.optimizer = None
        self.initialise(num_layers, num_nodes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # self.criterion = nn.BCELoss()
        

    def initialise(self, layers, neurons):
        nodes = [neurons]*self.num_layers
        hidden_layers = zip(nodes[:-1], nodes[1:])
        self.fcs = nn.ModuleList([nn.Linear(*self.input_dims, neurons)])
        self.fcs.extend([nn.Linear(h1, h2) for h1, h2 in hidden_layers])
        self.output = nn.Linear(neurons, self.output_dims)
        self.num_layers = layers
        self.num_nodes = neurons
        

    def forward(self, x):
        for layer in self.fcs:
            x = layer(x)
            x = F.relu(x)
        x = self.output(x)
        return x

    def add_neurons(self, num):

        # Getting the older weights of all layers
        weights = [fc.weight.data for fc in self.fcs]
        weights.append(self.output.weight.data)

        for index in range(len(self.fcs)):

            # make the new weights in and out of hidden layer you are adding
            # neurons to
            hl_input = T.zeros((num, self.fcs[index].weight.shape[1]))
            nn.init.xavier_uniform_(hl_input,
                                    gain=nn.init.calculate_gain('relu'))
            hl_output = T.zeros((weights[index+1].shape[0], num))
            nn.init.xavier_uniform_(hl_output,
                                    gain=nn.init.calculate_gain('relu'))

        # concatenate the old weights with the new weights
            new_wi = T.cat((self.fcs[index].weight, hl_input), dim=0)
            new_wo = T.cat((weights[index+1], hl_output), dim=1)

        # reset weight and grad variables to new size
            id1, id2 = self.fcs[index].weight.shape
            #print(id2, id1+num)
            self.fcs[index] = nn.Linear(id2, id1+num)

        # set the weight data to new values
            self.fcs[index].weight.data = new_wi.clone().detach().requires_grad_(True)
            if index == len(self.fcs)-1:
                # new_wo = T.cat((self.output.weight, hl_output), dim = 1)
                id1, id2 = self.output.weight.shape
                self.output = nn.Linear(id2+num, id1)
                self.output.weight.data = new_wo.clone().detach().requires_grad_(True)

            else:
                # new_wo = T.cat((self.fcs[index+1].weight, hl_output),
                # dim = 1)
                id1, id2 = self.fcs[index+1].weight.shape
                self.fcs[index+1] = nn.Linear(id2+num, id1)
                self.fcs[index+1].weight.data = new_wo.clone().detach().requires_grad_(True)

        self.num_nodes += num
        return [self.num_layers, self.num_nodes]

    def remove_neurons(self, num):
        
        # Getting the older weights of all layers
        weights = [fc.weight.data for fc in self.fcs]
        weights.append(self.output.weight.data)
        for index in range(self.num_layers):
            init_neurons = self.fcs[index].weight.shape[0]
            fin_neurons = init_neurons - num

            # Getting new weights by slicing the old weight tensor
            fin_neurons = max(fin_neurons, 1)
            new_wi = T.narrow(self.fcs[index].weight, 0, 0, fin_neurons)
            new_wo = T.narrow(weights[index+1], 1, 0, fin_neurons)

            # reset weight and grad variables to new size
            # set the weight data to new values
            id1, id2 = self.fcs[index].weight.shape
            self.fcs[index] = nn.Linear(id2, id1-num)
            # set the weight data to new values
            self.fcs[index].weight.data = new_wi.clone().detach().requires_grad_(True)

            if index == len(self.fcs)-1:
                id1, id2 = self.output.weight.shape
                self.output = nn.Linear(id2-num, id1)
                self.output.weight.data = new_wo.clone().detach().requires_grad_(True)
            else:
                id1, id2 = self.fcs[index+1].weight.shape
                self.fcs[index+1] = nn.Linear(id2-num, id1)
                self.fcs[index+1].weight.data = new_wo.clone().detach().requires_grad_(True)
        self.num_nodes -= num
        return [self.num_layers, self.num_nodes]

    def add_layers(self, num):
        last_hid_neurons = self.fcs[-1].weight.shape[0]
        new_hid_dims = [last_hid_neurons]*(num+1)
        new_hid_layers = zip(new_hid_dims[:-1], new_hid_dims[1:])
        self.fcs.extend([nn.Linear(h1, h2) for h1, h2 in new_hid_layers])
        self.num_layers += num
        
        return [self.num_layers, self.num_nodes]

    def remove_layers(self, num):
        x = len(self.fcs)-1
        for index in range(x, max(0,x-num), -1):
            self.fcs.__delitem__(index)
        self.num_layers -= num
        
        return [self.num_layers, self.num_nodes]

    def print_param(self):
        x = next(self.parameters()).data
        print(x)

    def train(self, trainloader):

        loss_list, acc_list = [], []
        for epochs in range(C.EPOCHS):
            correct = 0
            total = 0
            train_loss = 0
            for data, target in trainloader:   # print("Target = ",target[0].item())
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.forward(data.float())
                target = target.type(T.FloatTensor)
                loss = self.criterion(output, target.long().squeeze())
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update running training loss
                train_loss += loss.item()*data.size(0)
                total += target.size(0)

                # accuracy
                _, predicted = T.max(output.data, 1)
                correct += (predicted == target.squeeze()).sum().item()
            acc_list.append(100*correct/total)
            loss_list.append(train_loss/total)

        return mean(acc_list[-4:]), mean(loss_list[-4:])
    
    
    def test(self, testloader):
        correct = 0
        total = 0
        val_loss = 0
        with T.no_grad():
            for data, target in testloader:

            # Predict Output
                output = self.forward(data.float())

            # Calculate Loss
                target = target.view(-1)
                loss = self.criterion(output, target)
                val_loss += loss.item()*data.size(0)
            # Get predictions from the maximum value
                _, predicted = T.max(output.data, 1)

            # Total number of labels
                total += target.size(0)

        # Total correct predictions
                correct += (predicted == target.squeeze()).sum().item()

    # calculate average training loss and accuracy over an epoch
        val_loss = val_loss/len(testloader.dataset)
        accuracy = 100 * correct/float(total)
        return accuracy, val_loss

class Environment():
  def __init__(self, path='churn_modelling.csv', ):
    self.path = path
    self.dataset = Dataset(path = self.path)
    self.X, self.y = self.dataset.preprocess()
    self.X_train, self.X_test, self.y_train, self.y_test = self.dataset.split(self.X, self.y, fraction=0.2)
    self.X_train, self.X_test = self.dataset.scale(self.X_train, self.X_test)
    self.train_loader, self.test_loader = H.load(self.X_train, self.X_test,
                                                 self.y_train, self.y_test)
    
    self.input_dims = [self.X.shape[1]]
    self.output_dims = len(np.unique(self.y))
    
    self.model1 = TestModel(self.input_dims, self.output_dims, 0.005, 3, 8)
    self.model2 = TestModel(self.input_dims, self.output_dims, 0.005, 3, 8)

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
        self.model1.add_neurons(int(model1_action))
    else:
        self.model1.remove_neurons(int(-model1_action))
    
    model2_action = model1_layers - model2_layers
    if(model2_action >= 0):
        self.model2.add_layers(int(model2_action))
    else:
        self.model2.remove_layers(int(-model2_action))
    
    return [self.model2.num_layers, self.model1.num_nodes]
    
    
  
  def change_layers(self, action):
    
    if action >= 0:
        next_state = self.model1.add_layers(int(action))
    else:
        next_state = self.model1.remove_layers(-int(action))
    train_acc, train_loss = self.model1.train(self.train_loader)
    test_acc, test_loss = self.model1.test(self.test_loader)
    reward = H.reward(train_acc, train_loss,
                      test_acc, test_loss,
                      next_state, 0,
                      self.input_dims, self.output_dims)
    # reward = reward - punishment
    return next_state, reward
  
  def change_neurons(self, action):
    
    if action >= 0:
        next_state = self.model2.add_neurons(int(action))
    else:
        next_state = self.model2.remove_neurons(-int(action))
    train_acc, train_loss = self.model2.train(self.train_loader)
    test_acc, test_loss = self.model2.test(self.test_loader)
    reward = H.reward(train_acc, train_loss,
                      test_acc, test_loss,
                      next_state, 1,
                      self.input_dims, self.output_dims)
    return next_state, reward
  
 
  def seed(self):
    pass 

class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(CriticNetwork, self).__init__()

        self.lr = lr
        self.input_shape = input_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*input_dims,400)
        self.batch1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(400,300)
        self.batch2 = nn.LayerNorm(300)
        self.fc3 = nn.Linear(300, 1)
    
        self.action_value = nn.Linear(n_actions, 300)

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay=0.01)

        #self.initialize_weights_bias()
    def initialize_weights_bias(self):
    
        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)
        
        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        self.fc3.weight.data.uniform_(-0.003, 0.003)
        self.fc3.bias.data.uniform_(-0.003, 0.003)

        f4 = 1/np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

    def forward(self, state, action):
        x = self.fc1(state)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.batch2(x)
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(x,action_value))
        state_action_value = self.fc3(state_action_value)

        return state_action_value
    

class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, hd1_dims, hd2_dims,action_dim):
        super(ActorNetwork, self).__init__()

        self.lr = lr
        self.input_dims = input_dims
        self.hd1_dims = hd1_dims
        self.hd2_dims = hd2_dims
        self.action_dim = action_dim

        self.fc1 = nn.Linear(*self.input_dims, self.hd1_dims)
        self.fc2 = nn.Linear(self.hd1_dims, self.hd2_dims)
        self.fc3 = nn.Linear(self.hd2_dims, self.action_dim)

        self.nb1 = nn.LayerNorm(self.hd1_dims)
        self.nb2 = nn.LayerNorm(self.hd2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

        #self.initialize_weights_bias()
    def initialize_weights_bias(self):

        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)


        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.fc3.weight.data.uniform_(-f3,f3)
        self.fc3.bias.data.uniform_(-f3,f3)
        
    def forward(self, state):
        x = self.fc1(state)
        x = self.nb1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.nb2(x)
        x = F.relu(x)
        #x = self.fc3(x)
        x = T.tanh(self.fc3(x))
        
        return x.item()

class DDPGAgent():
    def __init__(self, alpha, beta, tau, input_dims, n_actions, hd1_dims = 400, hd2_dims = 300, mem_size = 1000000, gamma = 0.99, batch_size = 64):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
    
        self.localActor = ActorNetwork(self.alpha, input_dims, hd1_dims, hd2_dims, n_actions)
        self.localCritic = CriticNetwork(self.beta, input_dims, n_actions)
        self.targetActor = ActorNetwork(self.alpha, input_dims, hd1_dims, hd2_dims, n_actions)
        self.targetCritic = CriticNetwork(self.beta, input_dims, n_actions)

        self.replayBuffer = ReplayBuffer(mem_size, input_dims, n_actions)

        self.actionNoise = OrnsteinUhlenbeckActionNoise(mu = np.zeros(n_actions))
        
        self.update_parameter_weights(tau = 1)
        
    def choose_action(self, observation, agent_no):
        self.localActor.eval()
        state = torch.tensor([observation], dtype = T.float32)
        action = self.localActor.forward(state)
        noisy_action = action + torch.tensor(self.actionNoise(), dtype = torch.float32)

        self.localActor.train()
        final_action = C.MAX_ACTION[agent_no]*noisy_action.detach().numpy()[0]

        return (final_action, round(final_action))
    
    def store_transition(self, state, action, reward, next_state):
        self.replayBuffer.store_transition(state, action, reward, next_state)
    
    def learn(self):

        if self.replayBuffer.mem_cntr < self.batch_size:
            return

        states, actions, rewards, next_states = self.replayBuffer.sample_buffer(self.batch_size)

        states = torch.tensor(states, dtype = torch.float)
        actions = torch.tensor(actions, dtype = torch.float)
        rewards = torch.tensor(rewards, dtype = torch.float)
        next_states = torch.tensor(next_states, dtype = torch.float)
        #done = torch.tensor(done)

        Q = self.localCritic.forward(states, actions)
        target_actions = self.targetActor.forward(next_states)
        Q_prime = self.targetCritic.forward(next_states, target_actions)

        #Q_prime[done] = 0.0
        Q_prime = Q_prime.view(-1)
        y_prime = rewards + self.gamma*Q_prime
        y_prime = y_prime.view(self.batch_size, 1)


        self.localCritic.optimizer.zero_grad()
        criticLoss = F.mse_loss(y_prime, Q)
        criticLoss.backward()
        self.localCritic.optimizer.step()

        self.localActor.optimizer.zero_grad()
        actorLoss = -self.localCritic.forward(states, self.localActor.forward(states))
        actorLoss = torch.mean(actorLoss)
        actorLoss.backward()
        self.localActor.optimizer.step()
    
        self.update_parameter_weights()
        
    def update_parameter_weights(self, tau = None):
        if tau is None:
            tau = self.tau
        actor_dict = self.localActor.state_dict()
        target_actor_dict = self.targetActor.state_dict()

        critic_dict = self.localCritic.state_dict()
        target_critic_dict = self.targetCritic.state_dict()

        for key in target_actor_dict:
            target_actor_dict[key] = tau*target_actor_dict[key] + (1-tau)*actor_dict[key]

        self.targetActor.load_state_dict(target_actor_dict)

        for key in target_critic_dict:
            target_critic_dict[key] = tau*target_critic_dict[key] + (1-tau)*critic_dict[key]
    
        self.targetCritic.load_state_dict(target_critic_dict)


  
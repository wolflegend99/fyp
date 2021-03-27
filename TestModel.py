import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import constants as C
from statistics import mean


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
        self.initialise(num_layers, num_nodes, lr)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCELoss()
        

    def initialise(self, layers, neurons, lr):
        nodes = [neurons]*self.num_layers
        hidden_layers = zip(nodes[:-1], nodes[1:])
        self.fcs = nn.ModuleList([nn.Linear(*self.input_dims, neurons)])
        self.fcs.extend([nn.Linear(h1, h2) for h1, h2 in hidden_layers])
        self.output = nn.Linear(neurons, self.output_dims)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

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
            print(id2, id1+num)
            self.fcs[index] = nn.Linear(id2, id1+num)

        # set the weight data to new values
            self.fcs[index].weight.data = T.tensor(new_wi, requires_grad=True)

            if index == len(self.fcs)-1:
                # new_wo = T.cat((self.output.weight, hl_output), dim = 1)
                id1, id2 = self.output.weight.shape
                self.output = nn.Linear(id2+num, id1)
                self.output.weight.data = T.tensor(new_wo, requires_grad=True)

            else:
                # new_wo = T.cat((self.fcs[index+1].weight, hl_output),
                # dim = 1)
                id1, id2 = self.fcs[index+1].weight.shape
                self.fcs[index+1] = nn.Linear(id2+num, id1)
                self.fcs[index+1].weight.data = T.tensor(new_wo,
                                                         requires_grad=True)

        self.num_nodes += num

    def remove_neurons(self, num):

        # Getting the older weights of all layers
        weights = [fc.weight.data for fc in self.fcs]
        weights.append(self.output.weight.data)
        for index in range(self.num_layers):
            init_neurons = self.fcs[index].weight.shape[0]
            fin_neurons = init_neurons - num

            # Getting new weights by slicing the old weight tensor
            new_wi = T.narrow(self.fcs[index].weight, 0, 0, fin_neurons)
            new_wo = T.narrow(weights[index+1], 1, 0, fin_neurons)

            # reset weight and grad variables to new size
            # set the weight data to new values
            id1, id2 = self.fcs[index].weight.shape
            self.fcs[index] = nn.Linear(id2, id1-num)
            # set the weight data to new values
            self.fcs[index].weight.data = T.tensor(new_wi,
                                                      requires_grad=True)

            if index == len(self.fcs)-1:
                id1, id2 = self.output.weight.shape
                self.output = nn.Linear(id2-num, id1)
                self.output.weight.data = T.tensor(new_wo,
                                                       requires_grad=True)
            else:
                id1, id2 = self.fcs[index+1].weight.shape
                self.fcs[index+1] = nn.Linear(id2-num, id1)
                self.fcs[index+1].weight.data = T.tensor(new_wo,
                                                         requires_grad=True)
        self.num_nodes -= num

    def add_layers(self, num):
        last_hid_neurons = self.fcs[-1].weight.shape[0]
        new_hid_dims = [last_hid_neurons]*(num+1)
        new_hid_layers = zip(new_hid_dims[:-1], new_hid_dims[1:])
        self.fcs.extend([nn.Linear(h1, h2) for h1, h2 in new_hid_layers])
        self.num_layers += num

    def remove_layers(self, num):
        x = len(self.fcs)-1
        for index in range(x, x-num, -1):
            self.fcs.__delitem__(index)
        self.num_layers -= num

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
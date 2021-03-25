class Network(nn.Module):
  def __init__(self, lr, input_dims, hd_dims , output_dims):
    super(Network, self).__init__()

    self.lr = lr
    self.input_dims = input_dims
    self.hd_dims = hd_dims
    self.output_dims = output_dims

    hidden_layers = zip(self.hd_dims[:-1], self.hd_dims[1:])
    self.fcs = nn.ModuleList([nn.Linear(*self.input_dims, self.hd_dims[0])])
    self.fcs.extend([nn.Linear(h1, h2) for h1, h2 in hidden_layers])
    self.output = nn.Linear(self.hd_dims[-1], self.output_dims)

    self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
  
  def forward(self, x):
    for layers in self.fcs:
      x = F.relu(layers(x))
    
    x = self.output(x)
    return x
  
  def add_neurons(self, num, index):

    # Getting the older weights of all layers
    weights = [fc.weight.data for fc in self.fcs]
    weights.append(self.output.weight.data)

    # make the new weights in and out of hidden layer you are adding neurons to
    hl_input = T.zeros((num, weights[index].shape[1]))
    nn.init.xavier_uniform_(hl_input, gain=nn.init.calculate_gain('relu'))
    hl_output = T.zeros((weights[index+1].shape[0], num))
    nn.init.xavier_uniform_(hl_output, gain=nn.init.calculate_gain('relu'))

    # concatenate the old weights with the new weights
    new_wi = T.cat((weights[index], hl_input), dim = 0)
    new_wo = T.cat((weights[index+1], hl_output), dim = 1)

    # reset weight and grad variables to new size
    id1, id2 = self.fcs[index].weight.shape
    self.fcs[index] = nn.Linear(id2, id1+num)
    # set the weight data to new values
    self.fcs[index].weight.data = T.tensor(new_wi, requires_grad=True)

    if index == len(self.fcs)-1:
      id1, id2 = self.output.weight.shape
      self.output = nn.Linear(id2+num, id1)
      self.output.weight.data = T.tensor(new_wo , requires_grad=True)
    else:
      id1, id2 = self.fcs[index+1].weight.shape
      self.fcs[index+1] = nn.Linear(id2+num, id1)
      self.fcs[index+1].weight.data = T.tensor(new_wo , requires_grad=True)
  
  def remove_neurons(self, num, index):
    
    # Getting the older weights of all layers
    weights = [fc.weight.data for fc in self.fcs]
    weights.append(self.output.weight.data)

    init_neurons = weights[index].shape[0]
    fin_neurons = init_neurons - num

    #Getting new weights by slicing the old weight tensor
    new_wi = T.narrow(weights[index], 0, 0, fin_neurons)
    new_wo = T.narrow(weights[index+1], 1, 0, fin_neurons)

    # reset weight and grad variables to new size
    # set the weight data to new values
    id1, id2 = self.fcs[index].weight.shape
    self.fcs[index] = nn.Linear(id2, id1-num)
    # set the weight data to new values
    self.fcs[index].weight.data = T.tensor(new_wi, requires_grad=True)

    if index == len(self.fcs)-1:
      id1, id2 = self.output.weight.shape
      self.output = nn.Linear(id2-num, id1)
      self.output.weight.data = T.tensor(new_wo , requires_grad=True)
    else:
      id1, id2 = self.fcs[index+1].weight.shape
      self.fcs[index+1] = nn.Linear(id2-num, id1)
      self.fcs[index+1].weight.data = T.tensor(new_wo , requires_grad=True)

  def add_layers(self, num):
    last_hid_neurons = self.fcs[-1].weight.shape[0]
    new_hid_dims = [last_hid_neurons]*(num+1)

    new_hid_layers = zip(new_hid_dims[:-1], new_hid_dims[1:])
    self.fcs.extend([nn.Linear(h1, h2) for h1, h2 in new_hid_layers])
  
  def remove_layers(self, index):
    self.fcs.__delitem__(index)


  def print_param(self):
    x = next(self.parameters()).data
    print(x)

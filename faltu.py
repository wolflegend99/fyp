'''
import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from TestModel import TestModel
from dataset import Dataset
import helper as H


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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(10, 2))

    def forward(self, x):
        x = F.linear(x, self.weight)
        return x


# Create model and initialize all params
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
print(optimizer.state_dict())  # state is empty

criterion = nn.MSELoss()
x = torch.randn(1, 2)
target = torch.randn(1, 10)

# Train for a few epochs
for epoch in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print('Epoch {}, loss {}'.format(epoch, loss.item()))

# Store old id of parameters
old_id = id(model.weight)

# Add another input feature
with torch.no_grad():
    model.weight = nn.Parameter(
        torch.cat((model.weight, torch.randn(10, 1)), 1)
    )

# Store new id
new_id = id(model.weight)

# Get old state_dict and store all internals
opt_state_dict = optimizer.state_dict()
print(opt_state_dict)
step = opt_state_dict['state'][old_id]['step']
exp_avg = opt_state_dict['state'][old_id]['exp_avg']
exp_avg_sq = opt_state_dict['state'][old_id]['exp_avg_sq']

# Extend exp_avg_* to match new shape
exp_avg = torch.cat((exp_avg, torch.zeros(10, 1)), 1)
exp_avg_sq = torch.cat((exp_avg_sq, torch.zeros(10, 1)), 1)

# Delete old id from state_dict and update with new params and new id
del opt_state_dict['state'][old_id]
opt_state_dict['state'] = {
    new_id: {
        'step': step,
        'exp_avg': exp_avg,
        'exp_avg_sq': exp_avg_sq
    }
}
opt_state_dict['param_groups'][0]['params'].remove(old_id)
opt_state_dict['param_groups'][0]['params'].append(new_id)

# Create new optimizer and load state_dict with running estimates for old
# parameters
optimizer = optim.Adam(model.parameters(), lr=1e-1)
optimizer.load_state_dict(opt_state_dict)

# Continue training
x = torch.randn(1, 3)
for epoch in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print('Epoch {}, loss {}'.format(epoch, loss.item()))
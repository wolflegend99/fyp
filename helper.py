import constants as C
import torch
import numpy as np

def load(X_train, X_test, y_train, y_test):
    # Numpy to Tensor Conversion (Train Set)
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train).view(-1, 1)
    # Numpy to Tensor Conversion (Test Set)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test).view(-1, 1)

    # Make torch datasets from train and test sets
    train = torch.utils.data.TensorDataset(X_train, y_train)
    test = torch.utils.data.TensorDataset(X_test, y_test)
    # Create train and test data loaders
    train_loader = torch.utils.data.DataLoader(train, batch_size = C.BATCH_SIZE, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test, batch_size = C.BATCH_SIZE, shuffle = True)
    return train_loader, test_loader

def reward(train_acc, train_loss,
            test_acc, test_loss,
            x):
    return np.random.randint(1, 4)

def sample(data, limit=C.SAMPLE_SIZE):
    sample = data.sample(n=limit)
    return sample


def shuffle(data):
    return data.sample(frac=1)


def get_dimensions(self, data):
    return data.shape

class OrnsteinUhlenbeckActionNoise():
    def __init__(self, mu, sigma = 0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

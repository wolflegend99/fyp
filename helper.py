import constants as C
import torch


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


def sample(self, data, limit=C.SAMPLE_SIZE):
    sample = data.sample(n=limit)
    return sample


def shuffle(self, data):
    return data.sample(frac=1)


def get_dimensions(self, data):
    return data.shape

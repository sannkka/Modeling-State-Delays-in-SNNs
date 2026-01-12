from data_loaders import get_data_loaders
import numpy as np
import torch

data_name = 'shd'
batch_size = 256
include_val = True
time_window = 10000
path_server = '/mimer/NOBACKUP/groups/snn/Delays/'

train_loader, test_loader = get_data_loaders(
        time_window=time_window,
        batch_size=batch_size,
        cache_dir=path_server,
        num_workers=4,
        time_jitter=1.0,
        spatial_jitter=0.55,
        time_skew=1.2,
        max_drop_chunk=0.02,
        drop_event=0.1,
        noise=35,
)

n_bins = 5  # For binning input channels, so instead 700 we have 700 // n_bins
x_train, y_train, x_test, y_test = [], [], [], []

for step, (x, y) in enumerate(test_loader):
    print(f'nb time steps = {x.shape[1]}')
    x = x.squeeze()
    x = x.view(*x.shape[:-1], x.shape[-1] // n_bins, n_bins).sum(dim=-1)  # -> (A, B, C//5)
    y = torch.nn.functional.one_hot(y, num_classes=20)
    x_test.append(x)
    y_test.append(y)

for step, (x, y) in enumerate(train_loader):
    print(f'nb time steps = {x.shape[1]}')
    x = x.squeeze()
    x = x.view(*x.shape[:-1], x.shape[-1] // n_bins, n_bins).sum(dim=-1)  # -> (A, B, C//5)
    y = torch.nn.functional.one_hot(y, num_classes=20)
    x_train.append(x)
    y_train.append(y)

# Save dataset
dict = { # TODO doublecheck if torch cat/stack done on the right dim with this later change
    'x_train': torch.cat(x_train),
    'y_train': torch.cat(y_train),
    'x_test': torch.cat(x_test),
    'y_test': torch.cat(y_test),
}

with open(path_server + 'shd.pkl', 'wb') as f:
    pickle.dump(dict, f)


import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
from data_transforms import CutMix, TimeNeurons_mask_aug

path_to_encoded_data = "/mimer/NOBACKUP/groups/snn/Delays/shd.pkl"
load_device = 'cpu'
with open(path_to_encoded_data, 'rb') as f:
    dict = pickle.load(f)

x_train = dict['x_train'].to(device=load_device, dtype=torch.float32)
y_train = dict['y_train'].to(device=load_device)
x_test = dict['x_test'].to(device=load_device, dtype=torch.float32)
y_test = dict['y_test'].to(device=load_device)

# Crete datasets and dataloaders
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

aug_cutmix = CutMix(p=0.5)
aug_mask = TimeNeurons_mask_aug()

for step, (x, y) in enumerate(test_loader):
    print(f'nb time steps = {x.shape[1]}')
    x_org = x.clone()
    x1 = aug_mask(x)
    x1_org = x1.clone()
    x2, y2 = aug_cutmix(x, y.clone())
    k=0


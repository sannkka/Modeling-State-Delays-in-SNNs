from snn_with_delays import SNN
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
from data_transforms import CutMix, TimeNeurons_mask_aug

class SnnTrain():

    def __init__(self,
                 nb_input_neurons, nb_hidden_neurons, nb_output_neurons,
                 device,
                 neuron_type,
                 nb_epochs,
                 delay_order,
                 delay_parametrization,
                 delay_parametrization_trainable,
                 batch_size=128,
                 dropout=0.4,
                 lr=0.01,
                 w_decay=1e-5,
                 set_seed=0):
        # Hyperparameters
        self.device = device
        self.nb_input_neurons = nb_input_neurons
        self.nb_output_neurons = nb_output_neurons
        self.nb_hidden_neurons = nb_hidden_neurons
        self.set_seed = set_seed
        self.batch_size = batch_size

        self.net = SNN(nb_input_neurons=self.nb_input_neurons,
                       batch_size=self.batch_size,
                       neuron_type=neuron_type,
                       delay_order=delay_order,
                       delay_parametrization=delay_parametrization,
                       delay_parametrization_trainable=delay_parametrization_trainable,
                       layer_sizes=[self.nb_hidden_neurons, self.nb_hidden_neurons, self.nb_output_neurons],
                       dropout=dropout,
                       set_seed=set_seed,
                       ).to(self.device)

        # Other Hyper-parameters
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=w_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=nb_epochs, eta_min=1e-6)

        # Data Augmentation
        self.cutmix = CutMix(p=0.5)
        self.aug_mask = TimeNeurons_mask_aug()

    def train_one_epoch(self, train_loader):
        """
        This function trains the model with a single pass over the training split of the dataset.
        """
        self.net.train()

        # Loop over batches from train set
        for step, (x, y, *_) in enumerate(train_loader):

            # Dataloader uses cpu to allow pin memory
            x = x.to(self.device)
            y = y.to(self.device)

            # Extra Data Augmentation
            with torch.no_grad():
                x = self.aug_mask(x)
                x, y = self.cutmix(x, y)

            # Forward pass through network
            output, firing_rates = self.net(x)

            # Compute loss
            loss_val = self.loss_fn(output, y)

            # Back-propagate
            self.opt.zero_grad()
            loss_val.backward()
            self.opt.step()

        self.scheduler.step()

    def save_model_pre_train(self, file_path):

        if file_path is None:
            return

        # SAVE MODEL
        try:
            with open(file_path, 'rb') as f:
                saved_dict = pickle.load(f)
        except FileNotFoundError:
            saved_dict = {}

        self.net = self.net.to(device='cpu')

        # This is only when AdamW optimizer is used
        state_dict = self.opt.state_dict()['state']
        for i in range(len(state_dict)):
            self.opt.state_dict()['state'][i]['exp_avg'] = state_dict[i]['exp_avg'].to('cpu')
            self.opt.state_dict()['state'][i]['exp_avg_sq'] = state_dict[i]['exp_avg_sq'].to('cpu')

        saved_dict['model'] = self

        with open(file_path, 'wb') as f:
            pickle.dump(saved_dict, f)

    def test(self, loader):
        """
        Evaluate on the whole test set and return (accuracy_over_all_samples, mean_ce_loss_over_all_samples).
        """
        self.net.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Use a sum-reduction loss so we can normalize once at the end
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                output, firing_rates = self.net(x)

                # Sum loss over this batch
                loss = loss_fn(output, y)
                total_loss += loss.item()

                # Count correct predictions
                preds = torch.argmax(output, dim=1)
                y = torch.argmax(y, dim=1)
                total_correct += (preds == y).sum().item()

                total_samples += y.numel()

        # Global metrics over the entire dataset
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_acc, avg_loss

    def get_dataloaders(self):

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

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)

        val_loader = None
        return train_loader, test_loader, val_loader

    def train_network(self,
                    nb_epochs,
                    save_file_path,
                    validate_on_test=True):

        # Load Data
        train_loader, test_loader, val_loader = self.get_dataloaders(validate_on_test=validate_on_test)

        # TRAIN
        self.train_loss_hist = []
        self.train_accuracy_hist = []
        self.test_loss_hist = []
        self.test_accuracy_hist = []
        self.val_loss_hist = []
        self.val_accuracy_hist = []

        start_training = time.time()
        saving_time_cum = 0

        # Start training
        for e in range(1, nb_epochs + 1):
            self.train_one_epoch(train_loader=train_loader) # Train for 1 epoch

            # Printing
            start_saving = time.time()  # From here to end of loop time spend we consider it as outside-training time,
            if e % 10 == 0 or e == nb_epochs:
                print(f'Epochs {e}: \n ')

                # Find test accuracy and loss
                test_acc, test_loss = self.test(loader=test_loader)
                self.test_loss_hist.append(test_loss)
                self.test_accuracy_hist.append(test_acc)
                print(f'test acc = {test_acc*100:.4f} %, test loss = {test_loss:.4f} \n')

                # Find test accuracy and loss
                train_acc, train_loss = self.test(loader=train_loader)
                self.train_loss_hist.append(train_loss)
                self.train_accuracy_hist.append(train_acc)
                print(f'train acc = {train_acc*100:.4f} %, train loss = {train_loss:.4f} \n')

                # End of training
                end_saving = time.time()
                saving_time_cum += (end_saving - start_saving) / 60  # minutes

                if e == nb_epochs:
                    end_training = time.time()
                    total_training_time = (end_training - start_training) / 60 - saving_time_cum
                    self.total_train_time = total_training_time
                    print(f'Total train time = {total_training_time}')
                    self.save_model_pre_train(file_path=save_file_path)
                    print(f'Saved model at epoch {e}')


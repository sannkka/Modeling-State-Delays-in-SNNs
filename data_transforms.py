import numpy as np
import torch

class NumpyToTorch:
    def __call__(self, a):
        import numpy as np, torch
        if not a.flags['C_CONTIGUOUS']:
            a = np.ascontiguousarray(a)
        return torch.from_numpy(a).float()

class RemoveUnnecessaryInfo:
    # (T, H, W[, C]) -> (T, N)
    def __call__(self, sample):
        # Remove polarity
        sample = sample[:, 1, :]

        #  Bin frames spatially i.e. bin input channels
        n_bins = 5
        sample = sample.view(*sample.shape[:-1], sample.shape[-1] //n_bins, n_bins).sum(dim=-1)  # -> (A, B, C//5)
        return sample.contiguous()

class TimeNeurons_mask_aug:
    # Modfied from the H's TimeNeurons_mask_aug but per sample so it can be fitted in the transforms

    def __call__(self, sample):
        # From their paper:
        TN_mask_aug_proba = 0.65
        n_bins = 5
        n_inputs = 700 // n_bins
        time_step = 10
        max_delay = 250 // time_step
        time_mask_size = max_delay // 3
        neuron_mask_size = n_inputs // 5
        
        # Sample shape: (time, neurons)
        # Time mask
        if np.random.uniform() < TN_mask_aug_proba:
            mask_size = np.random.randint(0, time_mask_size)
            ind = np.random.randint(0, sample.shape[0] - time_mask_size)
            sample[ind:ind + mask_size, :] = 0

        # Neuron mask
        if np.random.uniform() < TN_mask_aug_proba:
            mask_size = np.random.randint(0, neuron_mask_size)
            ind = np.random.randint(0, sample.shape[2] - neuron_mask_size)
            sample[:, :, ind:ind + mask_size] = 0

        return sample


class CutMix:
    """
    Apply Spectrogram-CutMix augmentaiton which only cuts patch across time axis unlike
    typical Computer-Vision CutMix. Applies CutMix to one batch and its shifted version.

    """

    def __init__(self, p=0.5):
        self.p = p

    @torch.no_grad()
    def __call__(self, x, y):

        # x shape: (batch, time, neurons)
        # Go to L-1, no need to augment last sample in batch (for ease of coding)

        for i in range(x.shape[0] - 1):
            # other sample to cut from
            j = i + 1

            if np.random.uniform() < self.p:
                lam = np.random.uniform()
                cut_size = int(lam * x[j].shape[0])

                ind = np.random.randint(0, x[i].shape[0] - cut_size)

                x[i][ind:ind + cut_size, :] = x[j][ind:ind + cut_size, :]

                y[i] = (1 - lam) * y[i] + lam * y[j]

        return x, y

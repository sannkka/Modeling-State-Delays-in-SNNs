import torch
from pathlib import Path
from typing import Union
import tonic

DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')

def get_data_loaders(
        time_window,
        batch_size,
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
        num_workers: int = 0,
        time_jitter: float = 100,
        spatial_jitter: float = 1.0,
        max_drop_chunk: float = 0.1,
        noise: int = 100,
        drop_event: float = 0.1,
        time_skew: float = 1.1,):

    sensor_size = (700, 1, 1)

    train_transforms = tonic.transforms.Compose([
        tonic.transforms.DropEvent(p=drop_event),
        # DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
        # Jitter1D(sensor_size=sensor_size, var=spatial_jitter),
        tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
        tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
        tonic.transforms.UniformNoise(sensor_size=sensor_size, n=(0, noise)),

        tonic.transforms.MergePolarities(),
        tonic.transforms.ToFrame(sensor_size=(700, 1, 1), time_window=time_window),
    ])

    test_transforms = tonic.transforms.Compose([
        tonic.transforms.MergePolarities(),
        tonic.transforms.ToFrame(sensor_size=(700, 1, 1), time_window=time_window),
    ])

    train_data = tonic.datasets.SHD(save_to=cache_dir, train=True, transform=train_transforms)
    test_data = tonic.datasets.SHD(save_to=cache_dir, train=False, transform=test_transforms)

    collate_fn = tonic.collation.PadTensors()

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               collate_fn=collate_fn,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               prefetch_factor=4,
                                               pin_memory=True,  # <— add
                                               persistent_workers=(num_workers > 0),  # <— add
                                               )
    test_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=batch_size,
                                               collate_fn=collate_fn,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               prefetch_factor=4,
                                               pin_memory=True,  # <— add
                                               persistent_workers=(num_workers > 0),  # <— add
                                              )

    return train_loader, test_loader
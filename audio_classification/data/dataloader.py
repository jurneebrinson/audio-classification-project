from torch.utils.data import DataLoader
from audio_classification.data.dataset import SpectrogramDataset


def get_data_loader(root_dir, batch_size=4, shuffle=True, num_workers=0, transform=None):
    dataset = SpectrogramDataset(root_dir=root_dir, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

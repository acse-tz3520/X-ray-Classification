import torch

__all__ = ['CustomImageTensorDataset']


class CustomImageTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        """
        Args:
        data (Tensor): A tensor containing the data e.g. images
        targets (Tensor): A tensor containing all the labels
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx], self.targets[idx]
        sample = sample.view(-1, 224, 224)
        if self.transform:
            sample = self.transform(sample)

        return sample, label

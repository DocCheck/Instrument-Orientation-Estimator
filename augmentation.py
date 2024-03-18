import numpy as np
import torch
from torchvision import transforms




class GaussianNoise:
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:

        if torch.rand(1) < 0.25:
            noise = (torch.rand(size=(1, sample.shape[1], sample.shape[2])) < 0.8).int()
            noise = torch.where(noise == 0, 0.1, 1.0)
            if sample.shape[0] == 3:
                noise = torch.cat([noise, noise, noise], 0)
        else:
            noise = torch.ones(sample.shape)
        return sample * noise


class RndInvNoise:
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < 0.5:
            inv = transforms.functional.invert(sample)
            px, py = np.random.randint(low=2, high=sample.shape[1]), np.random.randint(low=2, high=sample.shape[2])
            w, h = np.random.randint(low=px, high=sample.shape[1]), np.random.randint(low=py, high=sample.shape[2])
            sample[:, px:px + w, py:py + w] = inv[:, px:px + w, py:py + w]
        return sample


def custom_augmentation_train():
    custom_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(-0.1, 0.1)),
        transforms.RandomResizedCrop(size=128, scale=(0.5, 1.5), ratio=(0.999, 1.001)),
        transforms.ToTensor(),
        GaussianNoise(),
        RndInvNoise(),
    ])
    return custom_transform


def custom_augmentation_test():
    custom_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.RandomResizedCrop(size=128, scale=(0.5, 1.5), ratio=(0.999, 1.001)),
        transforms.ToTensor(),
    ])
    return custom_transform
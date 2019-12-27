from importlib import import_module
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_loaders(opts):
    train_data, val_data = get_data(opts)
    train_loader = DataLoader(
        train_data, batch_size=opts.train_batch_size, shuffle=opts.shuffle,
        num_workers=opts.n_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_data, batch_size=opts.val_batch_size, shuffle=False,
        num_workers=opts.n_workers, pin_memory=False
    )
    opts.iter_per_epoch = len(train_loader)
    return train_loader, val_loader


def get_data(opts):
    if opts.dataset == 'mnist':
        t = transforms.Compose((
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)))
        )
        train_data = datasets.MNIST(
            root=opts.data_path, train=True, download=True, transform=t
        )
        val_data = datasets.MNIST(
            root=opts.data_path, train=False, download=True, transform=t
        )
    elif opts.dataset == 'cifar10':
        t = transforms.Compose((
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        )
        train_data = datasets.CIFAR10(
            root=opts.data_path, train=True, download=True, transform=t
        )
        val_data = datasets.CIFAR10(
            root=opts.data_path, train=False, download=True, transform=t
        )
    else:
        raise NotImplementedError()

    return train_data, val_data

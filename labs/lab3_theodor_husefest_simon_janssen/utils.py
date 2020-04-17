import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import json


def plot_images(X, labels):
    num_plots = X.shape[0]
    w_min, w_max = np.min(X), np.max(X)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for i in range(num_plots):
        plt.subplot(3, 3, i + 1)

        # Rescale the weights to be between 0 and 255
        img = 255.0 * (X[i, :, :, :].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(img.astype('uint8'))
        plt.axis('off')
        plt.title(classes[labels[i].item()])


def get_train_valid_loader(data_dir='data',
                           batch_size=64,
                           augment=False,
                           random_seed = 1,
                           valid_size=0.02,
                           shuffle=True,
                           show_sample=True,
                           num_workers=0,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def plot_weights(w, scaling=False):
    if w.shape[1] != 3:
        print('Cannot visualize filter with this dimensions: {}!'.format(w[0, :, :, :].shape))

    else:
        num_plots = w.shape[0]

        # plot positive weight entries
        plt.figure(num=None, figsize=(14, 6), dpi=80)
        for i in range(num_plots):
            plt.subplot(1, num_plots, i + 1)

            # Rescale the weights to be between 0 and 255
            img = w[i, :, :, :].squeeze().clip(0, 1)
            if scaling:
                img -= np.min(img)
                img /= np.max(img)
            img *= 255.
            plt.imshow(img.transpose().astype('uint8'))
            plt.axis('off')
            plt.title(str(i))
        plt.show()

        # plot negative weight entries
        plt.figure(num=None, figsize=(14, 6), dpi=80)
        for i in range(num_plots):
            plt.subplot(1, num_plots, i + 1)

            # Rescale the weights to be between 0 and 255
            img = w[i, :, :, :].squeeze().clip(-1, 0) * (-1)
            if scaling:
                img -= np.min(img)
                img /= np.max(img)
            img *= 255.
            plt.imshow(img.transpose().astype('uint8'))
            plt.axis('off')
            plt.title(str(i))
        plt.show()


def plot_gradient_flow(named_parameters):
    """
    Vizualize the gradient propagation through the neural net.

    """

    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:
        if (p.requires_grad) and ('bias' not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color='r')
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.1, lw=1, color='b')
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color='k')
    plt.xticks(range(len(ave_grads) + 1), layers, rotation='vertical')
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow")


def read_json(file_path):
    """Reading json data as dict.

    Args:
      file_path (str): File path.

    Return:
      json_dict (dict): Dictionary format of json data.

    """
    with open(file_path, 'r') as f:
        json_dict = json.load(f)
    return json_dict
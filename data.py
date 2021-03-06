import numpy as np
import torch
import torchvision
import os

from fake import FakeData

__all__ = ['loaders']

c10_classes = np.array([[0, 1, 2, 8, 9], [3, 4, 5, 6, 7]], dtype=np.int32)

def svhn_loaders(
    path,
    batch_size,
    num_workers,
    transform_train,
    transform_test,
    use_validation,
    val_size,
    shuffle_train=True,
):
    train_set = torchvision.datasets.SVHN(
        root=path, split="train", download=True, transform=transform_train
    )

    if use_validation:
        test_set = torchvision.datasets.SVHN(
            root=path, split="train", download=True, transform=transform_test
        )
        train_set.data = train_set.data[:-val_size]
        train_set.labels = train_set.labels[:-val_size]

        test_set.data = test_set.data[-val_size:]
        test_set.labels = test_set.labels[-val_size:]

    else:
        print("You are going to run models on the test set. Are you sure?")
        test_set = torchvision.datasets.SVHN(
            root=path, split="test", download=True, transform=transform_test
        )

    num_classes = 10

    return (
        {
            "train": torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "test": torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        },
        num_classes,
    )


def loaders(
    dataset,
    path,
    batch_size,
    num_workers,
    transform_train,
    transform_test,
    use_validation=True,
    val_size=5000,
    use_data_size = None,
    split_classes=None,
    shuffle_train=True,
    **kwargs
):

    path = os.path.join(path, dataset.lower())
    dl_dict = dict()
    
    if dataset == "SVHN":
        return svhn_loaders(
            path,
            batch_size,
            num_workers,
            transform_train,
            transform_test,
            use_validation,
            val_size,
        )
    elif dataset == 'CIFAR100Fake':
        ds = getattr(torchvision.datasets, 'CIFAR100')
    else:
        ds = getattr(torchvision.datasets, dataset)

    if dataset == "STL10":
        train_set = ds(
            root=path, split="train", download=True, transform=transform_train
        )
        num_classes = 10
        cls_mapping = np.array([0, 2, 1, 3, 4, 5, 7, 6, 8, 9])
        train_set.labels = cls_mapping[train_set.labels]
    elif dataset == "FakeData":
        train_set = FakeData(
            size=50000, image_size=(3, 32, 32), num_classes=100,
            transform=transform_train
        )
        num_classes=100
    else:
        train_set = ds(root=path, train=True, download=True, transform=transform_train)
        num_classes = max(train_set.targets) + 1

    if dataset == "CIFAR100Fake":
        print('Shuffling')
        from random import shuffle
        shuffle(train_set.targets)
        
    if use_data_size is not None:
        train_set.data = train_set.data[:use_data_size]
        train_set.targets = train_set.targets[:use_data_size]

    if use_validation:
        print(
            "Using train ("
            + str(len(train_set.data) - val_size)
            + ") + validation ("
            + str(val_size)
            + ")"
        )
        train_set.data = train_set.data[:-val_size]
        train_set.targets = train_set.targets[:-val_size]

        test_set = ds(root=path, train=True, download=True, transform=transform_test)
        test_set.train = False
        test_set.data = test_set.data[-val_size:]
        test_set.targets = test_set.targets[-val_size:]
        # delattr(test_set, 'data')
        # delattr(test_set, 'targets')
    else:
        print("You are going to run models on the test set. Are you sure?")
        if dataset == "STL10":
            test_set = ds(
                root=path, split="test", download=True, transform=transform_test
            )
            test_set.labels = cls_mapping[test_set.labels]
        elif dataset == "FakeData":
            test_set = FakeData(
                size=10000, image_size=(3, 32, 32), num_classes=100,
                transform=transform_train
            )
        else:
            test_set = ds(
                root=path, train=False, download=True, transform=transform_test
            )

    corrupt_train = kwargs.get("corrupt_train", None)
    if corrupt_train is not None and corrupt_train > 0:
        print("Train data corruption fraction:", corrupt_train)
        labels = np.array(train_set.targets)
        rs = np.random.RandomState(seed=228)
        mask = rs.rand(len(labels)) <= corrupt_train
        rnd_labels = rs.choice(num_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]
        assert len(train_set.targets) == len(labels)
        assert type(train_set.targets[0]) == type(labels[0])
        train_set.targets = labels
        
        return_train_subsets = kwargs.get("return_train_subsets", False)
        if return_train_subsets:
            corrupt_ids = np.arange(len(mask))[mask]
            normal_ids = np.arange(len(mask))[~mask]
            train_set_corrupt = torch.utils.data.Subset(train_set, corrupt_ids)
            train_set_normal = torch.utils.data.Subset(train_set, normal_ids)
            
            dl_dict.update({
                "train_corrupt": torch.utils.data.DataLoader(
                    train_set_corrupt,
                    batch_size=batch_size,
                    shuffle=shuffle_train,
                    num_workers=num_workers,
                    pin_memory=True,
                ),
                "train_normal": torch.utils.data.DataLoader(
                    train_set_normal,
                    batch_size=batch_size,
                    shuffle=shuffle_train,
                    num_workers=num_workers,
                    pin_memory=True,
                ),
            })

    if split_classes is not None:
        assert dataset == "CIFAR10"
        assert split_classes in {0, 1}

        print("Using classes:", end="")
        print(c10_classes[split_classes])
        train_mask = np.isin(train_set.targets, c10_classes[split_classes])
        train_set.data = train_set.data[train_mask, :]
        train_set.targets = np.array(train_set.targets)[train_mask]
        train_set.targets = np.where(
            train_set.targets[:, None] == c10_classes[split_classes][None, :]
        )[1].tolist()
        print("Train: %d/%d" % (train_set.data.shape[0], train_mask.size))

        test_mask = np.isin(test_set.targets, c10_classes[split_classes])
        print(test_set.data.shape, test_mask.shape)
        test_set.data = test_set.data[test_mask, :]
        test_set.targets = np.array(test_set.targets)[test_mask]
        test_set.targets = np.where(
            test_set.targets[:, None] == c10_classes[split_classes][None, :]
        )[1].tolist()
        print("Test: %d/%d" % (test_set.data.shape[0], test_mask.size))
        
    dl_dict.update({
        "train": torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    })

    return dl_dict, num_classes

from __future__ import annotations

import random
from typing import Any, Optional

TRAIN_VAL_TEST_PROTOCOL = "trainvaltest"


def stratified_trainvaltest_indices(targets: Any, *, seed: int = 42) -> tuple[list[int], list[int], list[int]]:
    by_class: dict[int, list[int]] = {}
    for index, target in enumerate(list(targets)):
        by_class.setdefault(int(target), []).append(int(index))

    rng = random.Random(int(seed))
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for label in sorted(by_class):
        indices = list(by_class[label])
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = int(round(n_total * 0.70))
        n_val = int(round(n_total * 0.20))
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train : n_train + n_val])
        test_indices.extend(indices[n_train + n_val :])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def split_existing_dataset_trainvaltest(
    train_source: Any,
    *,
    seed: int = 42,
    eval_source: Optional[Any] = None,
) -> dict[str, Any]:
    from torch.utils.data import Subset

    targets = getattr(train_source, "targets", None)
    if targets is None:
        targets = getattr(train_source, "labels", None)
    if targets is None:
        raise ValueError("Cannot build trainvaltest split: dataset has no targets/labels attribute")

    train_indices, val_indices, test_indices = stratified_trainvaltest_indices(
        targets,
        seed=int(seed),
    )
    eval_dataset = eval_source if eval_source is not None else train_source
    return {
        "protocol": TRAIN_VAL_TEST_PROTOCOL,
        "seed": int(seed),
        "train": Subset(train_source, train_indices),
        "reward_eval": Subset(eval_dataset, val_indices),
        "heldout_test": Subset(eval_dataset, test_indices),
    }


def split_train_val_dataset(
    train_source: Any,
    *,
    train_size: int,
    val_size: int,
    seed: int = 42,
) -> tuple[Any, Any]:
    from torch import Generator
    from torch.utils.data import random_split

    total_size = int(train_size) + int(val_size)
    if len(train_source) != total_size:
        raise ValueError(
            f"Cannot split dataset of length {len(train_source)} into {train_size}+{val_size}"
        )
    generator = Generator().manual_seed(int(seed))
    return random_split(train_source, [int(train_size), int(val_size)], generator=generator)


def build_classification_reward_split_datasets(
    *,
    train_source: Any,
    heldout_test_source: Any,
    train_size: int,
    val_size: int,
    seed: int = 42,
) -> dict[str, Any]:
    train_subset, val_subset = split_train_val_dataset(
        train_source,
        train_size=int(train_size),
        val_size=int(val_size),
        seed=int(seed),
    )
    return {
        "protocol": TRAIN_VAL_TEST_PROTOCOL,
        "seed": int(seed),
        "train": train_subset,
        "reward_eval": val_subset,
        "heldout_test": heldout_test_source,
    }


def build_cifar10_reward_split_datasets(
    *,
    root: str,
    transform: Any,
    download: bool,
    seed: int = 42,
) -> dict[str, Any]:
    from torchvision import datasets

    return build_classification_reward_split_datasets(
        train_source=datasets.CIFAR10(
            root=root,
            train=True,
            download=download,
            transform=transform,
        ),
        heldout_test_source=datasets.CIFAR10(
            root=root,
            train=False,
            download=download,
            transform=transform,
        ),
        train_size=45000,
        val_size=5000,
        seed=int(seed),
    )


def build_cifar10_split_datasets(
    *,
    root: str,
    train_transform: Any,
    eval_transform: Any,
    download: bool,
    seed: int = 42,
) -> dict[str, Any]:
    from torch.utils.data import Subset
    from torchvision import datasets

    train_source = datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=train_transform,
    )
    eval_source = datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=eval_transform,
    )
    train_subset, val_subset = split_train_val_dataset(
        train_source,
        train_size=45000,
        val_size=5000,
        seed=int(seed),
    )
    return {
        "protocol": TRAIN_VAL_TEST_PROTOCOL,
        "seed": int(seed),
        "train": train_subset,
        "reward_eval": Subset(eval_source, list(getattr(val_subset, "indices", []))),
        "heldout_test": datasets.CIFAR10(
            root=root,
            train=False,
            download=download,
            transform=eval_transform,
        ),
    }


def build_cifar100_reward_split_datasets(
    *,
    root: str,
    transform: Any,
    download: bool,
    seed: int = 42,
) -> dict[str, Any]:
    from torchvision import datasets

    return build_classification_reward_split_datasets(
        train_source=datasets.CIFAR100(
            root=root,
            train=True,
            download=download,
            transform=transform,
        ),
        heldout_test_source=datasets.CIFAR100(
            root=root,
            train=False,
            download=download,
            transform=transform,
        ),
        train_size=45000,
        val_size=5000,
        seed=int(seed),
    )


def build_cifar100_split_datasets(
    *,
    root: str,
    train_transform: Any,
    eval_transform: Any,
    download: bool,
    seed: int = 42,
) -> dict[str, Any]:
    from torch.utils.data import Subset
    from torchvision import datasets

    train_source = datasets.CIFAR100(
        root=root,
        train=True,
        download=download,
        transform=train_transform,
    )
    eval_source = datasets.CIFAR100(
        root=root,
        train=True,
        download=download,
        transform=eval_transform,
    )
    train_subset, val_subset = split_train_val_dataset(
        train_source,
        train_size=45000,
        val_size=5000,
        seed=int(seed),
    )
    return {
        "protocol": TRAIN_VAL_TEST_PROTOCOL,
        "seed": int(seed),
        "train": train_subset,
        "reward_eval": Subset(eval_source, list(getattr(val_subset, "indices", []))),
        "heldout_test": datasets.CIFAR100(
            root=root,
            train=False,
            download=download,
            transform=eval_transform,
        ),
    }


def build_imagenette_reward_split_datasets(
    *,
    root: str,
    transform: Any,
    download: bool,
    seed: int = 42,
) -> dict[str, Any]:
    from torchvision import datasets

    return build_classification_reward_split_datasets(
        train_source=datasets.Imagenette(
            root=root,
            split="train",
            download=download,
            transform=transform,
        ),
        heldout_test_source=datasets.Imagenette(
            root=root,
            split="val",
            download=download,
            transform=transform,
        ),
        train_size=7500,
        val_size=1969,
        seed=int(seed),
    )


def build_formal_reward_split_datasets(
    *,
    dataset_name: str,
    root: str,
    transform: Any,
    download: bool,
    seed: int = 42,
) -> dict[str, Any]:
    normalized_dataset = str(dataset_name or "").strip().lower().replace("_", "-")
    if normalized_dataset in {"cifar-10", "cifar10"}:
        return build_cifar10_reward_split_datasets(
            root=root,
            transform=transform,
            download=download,
            seed=int(seed),
        )
    if normalized_dataset in {"cifar-100", "cifar100"}:
        return build_cifar100_reward_split_datasets(
            root=root,
            transform=transform,
            download=download,
            seed=int(seed),
        )
    if normalized_dataset == "imagenette":
        return build_imagenette_reward_split_datasets(
            root=root,
            transform=transform,
            download=download,
            seed=int(seed),
        )
    raise ValueError(f"Unsupported formal dataset for reward split: {dataset_name!r}")

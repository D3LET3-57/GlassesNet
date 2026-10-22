"""
Data pipeline: CSVs, deterministic stratified folds, augmentations, loaders.
No NumPy is used anywhere in this module.
"""
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from src.config import Config

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


class GlassesDataset(Dataset):
    """Image dataset backed by a DataFrame (train/val/test)."""

    def __init__(self, df: pd.DataFrame, root_dir: str, transform,
                 is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.files = self.df["file"].tolist()
        self.ids = self.df["id"].astype(str).tolist()
        self.labels = (None if is_test
                       else self.df["glasses"].astype(int).tolist())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        path = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        if self.is_test:
            return image, self.ids[idx]
        return image, self.labels[idx]


def build_train_transform(cfg: Config):
    ops = []
    if cfg.aug_random_resized_crop:
        ops.append(T.RandomResizedCrop(
            cfg.image_size, scale=(cfg.aug_scale_min, 1.0),
            ratio=(0.8, 1.25),
        ))
    else:
        ops.append(T.Resize((cfg.image_size, cfg.image_size)))
    if cfg.aug_hflip:
        ops.append(T.RandomHorizontalFlip(p=0.5))
    if cfg.aug_rotation > 0:
        ops.append(T.RandomRotation(cfg.aug_rotation))
    if cfg.aug_color_jitter > 0:
        j = cfg.aug_color_jitter
        ops.append(T.ColorJitter(brightness=j, contrast=j,
                                 saturation=j, hue=min(j / 4, 0.05)))
    ops += [T.ToTensor(), T.Normalize(MEAN, STD)]
    if cfg.aug_random_erasing > 0:
        ops.append(T.RandomErasing(p=cfg.aug_random_erasing,
                                   scale=(0.02, 0.12), ratio=(0.5, 2.0)))
    return T.Compose(ops)


def build_eval_transform(image_size: int):
    return T.Compose([
        T.Resize(image_size + 32),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def load_train_df(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.train_csv)
    if cfg.limit_per_class > 0:
        parts = [
            g.sample(n=min(cfg.limit_per_class, len(g)), random_state=cfg.seed)
            for _, g in df.groupby("glasses")
        ]
        df = pd.concat(parts).sample(frac=1.0, random_state=cfg.seed)
    return df.reset_index(drop=True)


def stratified_folds(labels: torch.Tensor, num_folds: int, seed: int):
    """
    Deterministic stratified fold indices.

    Returns a list of (train_idx, val_idx) LongTensors.
    """
    g = torch.Generator().manual_seed(seed)
    fold_val = [[] for _ in range(num_folds)]
    for cls in labels.unique():
        idx = torch.nonzero(labels == cls, as_tuple=False).flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        for k, chunk in enumerate(torch.chunk(idx, num_folds)):
            fold_val[k].append(chunk)

    folds = []
    all_idx = torch.arange(len(labels))
    for k in range(num_folds):
        val_idx = torch.cat(fold_val[k]).sort().values
        mask = torch.ones(len(labels), dtype=torch.bool)
        mask[val_idx] = False
        train_idx = all_idx[mask]
        folds.append((train_idx, val_idx))
    return folds


def build_loaders(cfg: Config, df: pd.DataFrame | None = None, fold: int | None = None):
    """Return (train_loader, val_loader, train_df, val_df, train_idx, val_idx)."""
    if df is None:
        df = load_train_df(cfg)

    labels = torch.tensor(df["glasses"].tolist(), dtype=torch.long)
    folds = stratified_folds(labels, cfg.num_folds, cfg.split_seed)
    f = cfg.val_fold if fold is None else fold
    train_idx, val_idx = folds[f]

    train_df = df.iloc[train_idx.tolist()].reset_index(drop=True)
    val_df = df.iloc[val_idx.tolist()].reset_index(drop=True)

    train_ds = GlassesDataset(train_df, cfg.train_dir,
                              build_train_transform(cfg))
    val_ds = GlassesDataset(val_df, cfg.train_dir,
                            build_eval_transform(cfg.image_size))

    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    if cfg.num_workers > 0:
        common["prefetch_factor"] = 4

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader, train_df, val_df, train_idx, val_idx


def build_test_loader(cfg: Config, transform=None, batch_size: int | None = None,
                      num_workers: int | None = None):
    df = pd.read_csv(cfg.test_csv)
    if cfg.test_limit > 0:
        df = df.head(cfg.test_limit).reset_index(drop=True)
    ds = GlassesDataset(df, cfg.test_dir,
                        transform or build_eval_transform(cfg.image_size),
                        is_test=True)
    loader = DataLoader(
        ds,
        batch_size=batch_size or cfg.predict_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers if num_workers is None else num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    return loader, df

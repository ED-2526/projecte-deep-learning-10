import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from models.models import get_model


class DogBreedDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df.loc[idx, "image_path"]).convert("RGB")
        label = int(self.df.loc[idx, "encoded_breed"])
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(train=True, img_size=224, mode="standard"):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)

    if mode == "standard":
        # RandomResizedCrop pot agafar crops molt petits (8% de la imatge)
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])

    if mode == "pad":
        # Preserva l'aspect ratio: redimensiona fins que el costat llarg = img_size,
        # després omple amb negre. El gos sempre surt sencer.
        def pad_image(img):
            w, h = img.size
            scale = img_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            pad_w = img_size - new_w
            pad_h = img_size - new_h
            padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
            return transforms.functional.pad(img, padding, fill=0)

        if train:
            return transforms.Compose([
                transforms.Lambda(pad_image),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        return transforms.Compose([
            transforms.Lambda(pad_image),
            transforms.ToTensor(),
            normalize,
        ])

    if mode == "augmented":
        # Com standard però amb rotació i RandomErasing per forçar robustesa
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
            ])
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])

    raise ValueError(f"transform_mode desconegut: {mode}")


def make_loader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        pin_memory=True, num_workers=2,
    )


def make(config, device="cuda"):
    data_dir = config.data_dir
    img_size = config.img_size

    labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    encoder = LabelEncoder()
    labels_df["encoded_breed"] = encoder.fit_transform(labels_df["breed"])
    labels_df["image_path"] = labels_df["id"].apply(
        lambda x: os.path.join(data_dir, "train", f"{x}.jpg")
    )

    train_df, val_df = train_test_split(
        labels_df, test_size=0.2, random_state=42, stratify=labels_df["breed"]
    )

    mode = config.transform_mode
    num_classes = len(encoder.classes_)
    train_dataset = DogBreedDataset(train_df, transform=get_transforms(True,  img_size, mode))
    val_dataset   = DogBreedDataset(val_df,   transform=get_transforms(False, img_size, mode))

    train_loader = make_loader(train_dataset, config.batch_size, shuffle=True)
    val_loader   = make_loader(val_dataset,   config.batch_size, shuffle=False)

    model     = get_model(config.architecture, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
    )

    return model, train_loader, val_loader, criterion, optimizer, encoder

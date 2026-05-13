import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from models.models import get_model


BREED_TO_GROUP = {
    'airedale': 'terrier_small', 'australian_terrier': 'terrier_small',
    'bedlington_terrier': 'terrier_small', 'border_terrier': 'terrier_small',
    'cairn': 'terrier_small', 'dandie_dinmont': 'terrier_small',
    'irish_terrier': 'terrier_small', 'kerry_blue_terrier': 'terrier_small',
    'lakeland_terrier': 'terrier_small', 'norfolk_terrier': 'terrier_small',
    'norwich_terrier': 'terrier_small', 'scotch_terrier': 'terrier_small',
    'sealyham_terrier': 'terrier_small', 'silky_terrier': 'terrier_small',
    'soft-coated_wheaten_terrier': 'terrier_small',
    'west_highland_white_terrier': 'terrier_small',
    'wire-haired_fox_terrier': 'terrier_small', 'yorkshire_terrier': 'terrier_small',
    'american_staffordshire_terrier': 'terrier_bull',
    'staffordshire_bullterrier': 'terrier_bull',
    'bull_mastiff': 'terrier_bull', 'boxer': 'terrier_bull',
    'afghan_hound': 'sighthound', 'borzoi': 'sighthound',
    'saluki': 'sighthound', 'whippet': 'sighthound',
    'italian_greyhound': 'sighthound', 'scottish_deerhound': 'sighthound',
    'irish_wolfhound': 'sighthound',
    'basset': 'scent_hound', 'beagle': 'scent_hound',
    'bloodhound': 'scent_hound', 'bluetick': 'scent_hound',
    'black-and-tan_coonhound': 'scent_hound', 'walker_hound': 'scent_hound',
    'redbone': 'scent_hound', 'otterhound': 'scent_hound',
    'english_foxhound': 'scent_hound', 'rhodesian_ridgeback': 'scent_hound',
    'chesapeake_bay_retriever': 'retriever', 'curly-coated_retriever': 'retriever',
    'flat-coated_retriever': 'retriever', 'golden_retriever': 'retriever',
    'labrador_retriever': 'retriever',
    'blenheim_spaniel': 'spaniel', 'brittany_spaniel': 'spaniel',
    'clumber': 'spaniel', 'cocker_spaniel': 'spaniel',
    'english_springer': 'spaniel', 'irish_water_spaniel': 'spaniel',
    'sussex_spaniel': 'spaniel', 'welsh_springer_spaniel': 'spaniel',
    'german_short-haired_pointer': 'pointer_setter', 'vizsla': 'pointer_setter',
    'weimaraner': 'pointer_setter', 'english_setter': 'pointer_setter',
    'gordon_setter': 'pointer_setter', 'irish_setter': 'pointer_setter',
    'chow': 'spitz', 'keeshond': 'spitz', 'pomeranian': 'spitz',
    'samoyed': 'spitz', 'siberian_husky': 'spitz', 'malamute': 'spitz',
    'eskimo_dog': 'spitz', 'norwegian_elkhound': 'spitz', 'schipperke': 'spitz',
    'border_collie': 'herding', 'collie': 'herding',
    'shetland_sheepdog': 'herding', 'old_english_sheepdog': 'herding',
    'kelpie': 'herding', 'malinois': 'herding', 'german_shepherd': 'herding',
    'groenendael': 'herding', 'bouvier_des_flandres': 'herding',
    'bernese_mountain_dog': 'mountain', 'greater_swiss_mountain_dog': 'mountain',
    'appenzeller': 'mountain', 'entlebucher': 'mountain',
    'saint_bernard': 'mountain', 'great_pyrenees': 'mountain',
    'newfoundland': 'mountain', 'leonberg': 'mountain', 'kuvasz': 'mountain',
    'rottweiler': 'guardian', 'doberman': 'guardian',
    'great_dane': 'guardian', 'tibetan_mastiff': 'guardian',
    'chihuahua': 'toy', 'maltese_dog': 'toy', 'papillon': 'toy',
    'pekinese': 'toy', 'pug': 'toy', 'shih-tzu': 'toy',
    'japanese_spaniel': 'toy', 'lhasa': 'toy',
    'toy_poodle': 'poodle', 'miniature_poodle': 'poodle', 'standard_poodle': 'poodle',
    'affenpinscher': 'pinscher', 'miniature_pinscher': 'pinscher',
    'giant_schnauzer': 'pinscher', 'standard_schnauzer': 'pinscher',
    'miniature_schnauzer': 'pinscher',
    'cardigan': 'corgi', 'pembroke': 'corgi',
    'basenji': 'primitive', 'african_hunting_dog': 'primitive',
    'dhole': 'primitive', 'dingo': 'primitive', 'mexican_hairless': 'primitive',
}


class DogBreedDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img         = Image.open(self.df.loc[idx, "image_path"]).convert("RGB")
        breed_label = int(self.df.loc[idx, "encoded_breed"])
        group_label = int(self.df.loc[idx, "encoded_group"])
        if self.transform:
            img = self.transform(img)
        return img, breed_label, group_label


def _pad_image(img):
    w, h   = img.size
    max_wh = max(w, h)
    pad_w  = (max_wh - w) // 2
    pad_h  = (max_wh - h) // 2
    return transforms.functional.pad(img, (pad_w, pad_h, max_wh - w - pad_w, max_wh - h - pad_h), fill=0)


def _train_transform(img_size):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Lambda(_pad_image),
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _val_transform(img_size):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Lambda(_pad_image),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def make(config, device="cuda"):
    data_dir = config.data_dir
    img_size = config.img_size

    labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"))

    encoder = LabelEncoder()
    labels_df["encoded_breed"] = encoder.fit_transform(labels_df["breed"])

    labels_df["group"]         = labels_df["breed"].map(BREED_TO_GROUP).fillna("unknown")
    group_encoder              = LabelEncoder()
    labels_df["encoded_group"] = group_encoder.fit_transform(labels_df["group"])

    labels_df["image_path"] = labels_df["id"].apply(
        lambda x: os.path.join(data_dir, "train", f"{x}.jpg")
    )

    train_df, val_df = train_test_split(
        labels_df, test_size=config.val_split, random_state=42, stratify=labels_df["breed"]
    )

    num_classes = len(encoder.classes_)
    num_groups  = len(group_encoder.classes_)

    train_dataset = DogBreedDataset(train_df, transform=_train_transform(img_size))
    val_dataset   = DogBreedDataset(val_df,   transform=_val_transform(img_size))

    train_labels   = train_df["encoded_breed"].values
    class_counts   = train_df["encoded_breed"].value_counts().sort_index().values.astype(float)
    sample_weights = torch.DoubleTensor(1.0 / class_counts[train_labels])
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              sampler=sampler, pin_memory=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size,
                              shuffle=False,  pin_memory=True, num_workers=2)

    model = get_model(num_classes, num_groups, dropout=config.dropout,
                      pretrained=getattr(config, "pretrained", True)).to(device)

    criterion       = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing_eps)
    group_criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing_eps)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    return model, train_loader, val_loader, criterion, group_criterion, optimizer, encoder, val_df

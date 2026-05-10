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


# Mapping from the 120 Stanford-Dogs breed names to 16 visually coherent groups.
# Used to build the auxiliary group labels that drive hierarchical supervision.
# Breeds not listed here are assigned to "unknown" (handled gracefully in make()).
BREED_TO_GROUP = {
    # 1. Small terriers: wiry coat, pointed muzzle
    'airedale': 'terrier_small', 'australian_terrier': 'terrier_small',
    'bedlington_terrier': 'terrier_small', 'border_terrier': 'terrier_small',
    'cairn': 'terrier_small', 'dandie_dinmont': 'terrier_small',
    'irish_terrier': 'terrier_small', 'kerry_blue_terrier': 'terrier_small',
    'lakeland_terrier': 'terrier_small', 'norfolk_terrier': 'terrier_small',
    'norwich_terrier': 'terrier_small', 'scotch_terrier': 'terrier_small',
    'sealyham_terrier': 'terrier_small', 'silky_terrier': 'terrier_small',
    'soft-coated_wheaten_terrier': 'terrier_small',
    'west_highland_white_terrier': 'terrier_small',
    'wire-haired_fox_terrier': 'terrier_small',
    'yorkshire_terrier': 'terrier_small',
    # 2. Bull terriers: broad head, muscular body
    'american_staffordshire_terrier': 'terrier_bull',
    'staffordshire_bullterrier': 'terrier_bull',
    'bull_mastiff': 'terrier_bull', 'boxer': 'terrier_bull',
    # 3. Sighthounds: slim build, long legs, deep chest
    'afghan_hound': 'sighthound', 'borzoi': 'sighthound',
    'saluki': 'sighthound', 'whippet': 'sighthound',
    'italian_greyhound': 'sighthound', 'scottish_deerhound': 'sighthound',
    'irish_wolfhound': 'sighthound',
    # 4. Scent hounds: long pendulous ears, heavy bone
    'basset': 'scent_hound', 'beagle': 'scent_hound',
    'bloodhound': 'scent_hound', 'bluetick': 'scent_hound',
    'black-and-tan_coonhound': 'scent_hound', 'walker_hound': 'scent_hound',
    'redbone': 'scent_hound', 'otterhound': 'scent_hound',
    'english_foxhound': 'scent_hound', 'rhodesian_ridgeback': 'scent_hound',
    # 5. Retrievers: medium-large, water-resistant coat
    'chesapeake_bay_retriever': 'retriever', 'curly-coated_retriever': 'retriever',
    'flat-coated_retriever': 'retriever', 'golden_retriever': 'retriever',
    'labrador_retriever': 'retriever',
    # 6. Spaniels: wavy coat, floppy ears
    'blenheim_spaniel': 'spaniel', 'brittany_spaniel': 'spaniel',
    'clumber': 'spaniel', 'cocker_spaniel': 'spaniel',
    'english_springer': 'spaniel', 'irish_water_spaniel': 'spaniel',
    'sussex_spaniel': 'spaniel', 'welsh_springer_spaniel': 'spaniel',
    # 7. Pointers and setters: lean, athletic gun dogs
    'german_short-haired_pointer': 'pointer_setter', 'vizsla': 'pointer_setter',
    'weimaraner': 'pointer_setter', 'english_setter': 'pointer_setter',
    'gordon_setter': 'pointer_setter', 'irish_setter': 'pointer_setter',
    # 8. Spitz / Nordic: double coat, curled tail, erect ears
    'chow': 'spitz', 'keeshond': 'spitz', 'pomeranian': 'spitz',
    'samoyed': 'spitz', 'siberian_husky': 'spitz', 'malamute': 'spitz',
    'eskimo_dog': 'spitz', 'norwegian_elkhound': 'spitz', 'schipperke': 'spitz',
    # 9. Herding dogs: agile, medium size, alert expression
    'border_collie': 'herding', 'collie': 'herding',
    'shetland_sheepdog': 'herding', 'old_english_sheepdog': 'herding',
    'kelpie': 'herding', 'malinois': 'herding', 'german_shepherd': 'herding',
    'groenendael': 'herding', 'bouvier_des_flandres': 'herding',
    # 10. Mountain dogs: large, heavy, long fluffy coat
    'bernese_mountain_dog': 'mountain', 'greater_swiss_mountain_dog': 'mountain',
    'appenzeller': 'mountain', 'entlebucher': 'mountain',
    'saint_bernard': 'mountain', 'great_pyrenees': 'mountain',
    'newfoundland': 'mountain', 'leonberg': 'mountain', 'kuvasz': 'mountain',
    # 11. Large guardian dogs: muscular, short coat
    'rottweiler': 'guardian', 'doberman': 'guardian',
    'great_dane': 'guardian', 'tibetan_mastiff': 'guardian',
    # 12. Toy / companion dogs: very small, flat face
    'chihuahua': 'toy', 'maltese_dog': 'toy', 'papillon': 'toy',
    'pekinese': 'toy', 'pug': 'toy', 'shih-tzu': 'toy',
    'japanese_spaniel': 'toy', 'lhasa': 'toy',
    # 13. Poodles: curly coat, various sizes
    'toy_poodle': 'poodle', 'miniature_poodle': 'poodle', 'standard_poodle': 'poodle',
    # 14. Pinschers and schnauzers: square build, wiry beard
    'affenpinscher': 'pinscher', 'miniature_pinscher': 'pinscher',
    'giant_schnauzer': 'pinscher', 'standard_schnauzer': 'pinscher',
    'miniature_schnauzer': 'pinscher',
    # 15. Corgis: low body, large ears
    'cardigan': 'corgi', 'pembroke': 'corgi',
    # 16. Primitive / wild types: ancient breeds, varied appearance
    'basenji': 'primitive', 'african_hunting_dog': 'primitive',
    'dhole': 'primitive', 'dingo': 'primitive', 'mexican_hairless': 'primitive',
}


class DogBreedDataset(Dataset):
    def __init__(self, df, transform=None):
        # reset_index ensures .loc[idx] works on 0-based integers after train/val split
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image and convert to RGB (some JPEGs are grayscale or RGBA)
        img         = Image.open(self.df.loc[idx, "image_path"]).convert("RGB")
        breed_label = int(self.df.loc[idx, "encoded_breed"])   # 0 … 119
        group_label = int(self.df.loc[idx, "encoded_group"])   # 0 … num_groups-1

        if self.transform:
            img = self.transform(img)   # returns tensor or tuple-of-tensors (TTA)

        return img, breed_label, group_label


def get_transforms(train=True, img_size=224, mode="standard"):
    """
    Returns a torchvision transform pipeline for the given split and augmentation mode.
    All modes use ImageNet mean/std normalisation so the pretrained backbone expectations
    are satisfied.
    """
    mean      = [0.485, 0.456, 0.406]
    std       = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)

    if mode == "standard":
        # Training: random crop + random horizontal flip + mild colour jitter
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(img_size),          # random scale/crop
                transforms.RandomHorizontalFlip(),               # mirror with p=0.5
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        # Validation: deterministic centre crop (no randomness → reproducible metrics)
        return transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])

    if mode == "pad":
        # Preserves aspect ratio: resize so the long side = img_size, then zero-pad.
        # Ensures the full dog is always visible (no cropping artefacts).
        def pad_image(img):
            w, h   = img.size
            scale  = img_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img    = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            pad_w  = img_size - new_w
            pad_h  = img_size - new_h
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
        # Stronger augmentation: same aggressive crop as "standard" (scale 8-100%) PLUS
        # rotation and RandomErasing.  The key mistake to avoid is restricting the crop
        # to scale=(0.5, 1.0) — that makes training too easy and causes overfitting.
        # RandomErasing simulates occlusion (collar, leash, partially hidden dog) and
        # forces the model to rely on shape/structure rather than a single body region.
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(img_size),          # scale=(0.08,1.0) — same as standard
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),                   # rotational robustness
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),  # erase random patch
            ])
        return transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])

    raise ValueError(f"transform_mode desconegut: {mode}")


def make_loader(dataset, batch_size, shuffle=True, sampler=None):
    # sampler and shuffle are mutually exclusive in PyTorch DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        pin_memory=True,    # faster host→GPU transfer
        num_workers=2,
    )


def make(config, device="cuda"):
    """
    Build all objects needed for training: model, dataloaders, losses, optimiser.

    Returns
    -------
    model          : DualHeadModel on `device`
    train_loader   : DataLoader for the 80% training split
    val_loader     : DataLoader for the 20% validation split (standard transforms)
    criterion      : CrossEntropyLoss with label smoothing for the breed head
    group_criterion: CrossEntropyLoss with label smoothing for the group head
    optimizer      : Adam, filtering to only trainable parameters
    encoder        : LabelEncoder for breed names (encoder.classes_[i] = breed name)
    val_df         : raw validation DataFrame (passed to test() for TTA)
    """
    data_dir = config.data_dir
    img_size = config.img_size

    labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"))

    # --- Breed labels (0 … 119) ---
    encoder = LabelEncoder()
    labels_df["encoded_breed"] = encoder.fit_transform(labels_df["breed"])

    # --- Group labels (0 … num_groups-1) ---
    # Map each breed to its visual group; unrecognised breeds → "unknown"
    labels_df["group"]         = labels_df["breed"].map(BREED_TO_GROUP).fillna("unknown")
    group_encoder              = LabelEncoder()
    labels_df["encoded_group"] = group_encoder.fit_transform(labels_df["group"])

    labels_df["image_path"] = labels_df["id"].apply(
        lambda x: os.path.join(data_dir, "train", f"{x}.jpg")
    )

    # Stratified split → every breed has the same train/val ratio
    train_df, val_df = train_test_split(
        labels_df, test_size=0.2, random_state=42, stratify=labels_df["breed"]
    )

    mode        = config.transform_mode
    num_classes = len(encoder.classes_)
    num_groups  = len(group_encoder.classes_)

    train_dataset = DogBreedDataset(train_df, transform=get_transforms(True,  img_size, mode))
    val_dataset   = DogBreedDataset(val_df,   transform=get_transforms(False, img_size, mode))

    # Weighted random sampling: breeds with fewer training examples (or that the model
    # historically struggles with, like Eskimo Dog) are oversampled so each epoch sees
    # a more balanced distribution.  Weight = 1 / class_frequency.
    train_labels   = train_df["encoded_breed"].values
    class_counts   = train_df["encoded_breed"].value_counts().sort_index().values.astype(float)
    sample_weights = torch.DoubleTensor(1.0 / class_counts[train_labels])
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,   # sample with replacement so every epoch has len(train) steps
    )

    train_loader = make_loader(train_dataset, config.batch_size, sampler=sampler)
    val_loader   = make_loader(val_dataset,   config.batch_size, shuffle=False)

    model = get_model(config.architecture, num_classes, num_groups).to(device)

    # Label smoothing (ε=0.1): instead of the one-hot target [0,…,0,1,0,…,0], the
    # target becomes [ε/K,…,ε/K, 1−ε+ε/K, ε/K,…,ε/K].  This prevents the model
    # from becoming overconfident, which is especially useful in fine-grained tasks
    # where class boundaries are genuinely soft (e.g. golden vs flat-coated retriever).
    criterion       = nn.CrossEntropyLoss(label_smoothing=0.1)
    group_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW adds L2 weight decay directly to the parameter update (not via the gradient),
    # which is more correct than Adam + L2 and reduces overfitting in the fine-tuning phase.
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=1e-4,
    )

    return model, train_loader, val_loader, criterion, group_criterion, optimizer, encoder, val_df

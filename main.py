import random
import wandb
import numpy as np
import torch

from train import train
from test import test
from utils.utils import make
from models.models import unfreeze_top_layers

# Fix all random seeds for reproducibility across runs
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


_TRANSFORM_ABBREV = {"standard": "ST", "pad": "PAD", "augmented": "AUG"}

def _run_name(cfg: dict) -> str:
    arch  = cfg["architecture"]
    mode  = _TRANSFORM_ABBREV.get(cfg["transform_mode"], cfg["transform_mode"].upper())
    ft    = "FT" if cfg["finetune"] else "N"
    ep    = cfg["epochs"]
    return f"{arch}_{mode}_{ft}_{ep}"


def model_pipeline(cfg: dict) -> None:
    with wandb.init(project="dog-breed-identification", name=_run_name(cfg), config=cfg):
        config = wandb.config

        model, train_loader, val_loader, criterion, optimizer, encoder = make(config, device=device)
        wandb.watch(model, log="all", log_freq=50)

        # Phase 1: frozen backbone, only head trains
        train(model, train_loader, val_loader, criterion, optimizer, config,
              device=device, phase=1)

        # Phase 2: unfreeze last conv block, discriminative learning rates
        if config.finetune:
            unfreeze_top_layers(model, config.architecture)
            if config.architecture in ("resnet50", "resnet18"):
                optimizer_ft = torch.optim.Adam([
                    {"params": model.layer4.parameters(), "lr": config.learning_rate * 0.1},
                    {"params": model.fc.parameters(),     "lr": config.learning_rate},
                ])
            else:
                optimizer_ft = torch.optim.Adam([
                    {"params": model.features[-3:].parameters(), "lr": config.learning_rate * 0.1},
                    {"params": model.classifier.parameters(),    "lr": config.learning_rate},
                ])
            train(model, train_loader, val_loader, criterion, optimizer_ft, config,
                  device=device, epochs=config.finetune_epochs, phase=2)

        test(model, val_loader, device=device)

    return model


if __name__ == "__main__":
    wandb.login()

    config = dict(
        epochs=20,
        patience=5,
        finetune=True, #false per al resnet50, no millora
        finetune_epochs=15,
        batch_size=32,
        learning_rate=1e-3,
        architecture="resnet50",       # "resnet18" | "resnet50" | "efficientnet_b0"
        transform_mode="standard",     # "standard" | "pad" | "augmented"
        data_dir="../../dades",
        img_size=224,
    )
    model = model_pipeline(config)

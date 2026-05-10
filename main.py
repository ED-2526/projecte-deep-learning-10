import os
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
    arch = cfg["architecture"]
    mode = _TRANSFORM_ABBREV.get(cfg["transform_mode"], cfg["transform_mode"].upper())
    ft   = "FT" if cfg["finetune"] else "N"
    ep   = cfg["epochs"]
    return f"{arch}_{mode}_{ft}_{ep}"


def model_pipeline(cfg: dict) -> None:
    with wandb.init(project="projecte-deep-learning-10-main", name=_run_name(cfg), config=cfg):
        config = wandb.config

        # make() builds: dual-head model, dataloaders, losses, optimiser, encoders
        (model, train_loader, val_loader,
         criterion, group_criterion,
         optimizer, encoder, val_df) = make(config, device=device)

        # Watch once: logs gradients and weights of every parameter every 50 batches.
        # Called here (not inside train()) to avoid a double-watch error on finetune runs.
        wandb.watch(model, log="all", log_freq=50)

        # ---- Phase 1: frozen backbone, only breed head + group head train ----
        train(model, train_loader, val_loader, criterion, group_criterion, optimizer,
              config, device=device, phase=1)

        # ---- Phase 2: unfreeze last conv block with discriminative learning rates ----
        # The backbone layers get a 10× smaller LR than the heads: they are already
        # well-trained from ImageNet and need only fine adjustments, while the heads
        # still benefit from a full learning rate.
        if config.finetune:
            unfreeze_top_layers(model, config.architecture)

            if config.architecture in ("resnet50", "resnet18"):
                optimizer_ft = torch.optim.AdamW([
                    {"params": model.layer4.parameters(),     "lr": config.learning_rate * 0.1},
                    {"params": model.fc.parameters(),         "lr": config.learning_rate},
                    {"params": model.group_head.parameters(), "lr": config.learning_rate},
                ], weight_decay=1e-4)
            else:
                optimizer_ft = torch.optim.AdamW([
                    {"params": model.features[-3:].parameters(), "lr": config.learning_rate * 0.1},
                    {"params": model.fc.parameters(),            "lr": config.learning_rate},
                    {"params": model.group_head.parameters(),    "lr": config.learning_rate},
                ], weight_decay=1e-4)

            train(model, train_loader, val_loader, criterion, group_criterion, optimizer_ft,
                  config, device=device, epochs=config.finetune_epochs, phase=2)

        # ---- Final evaluation with TTA + confusion matrix + per-class analysis ----
        test(model, val_df, config, device=device, class_names=list(encoder.classes_))

    return model


if __name__ == "__main__":
    wandb.login()

    config = dict(
        epochs=30,
        patience=10,
        finetune=True,
        finetune_epochs=15,
        batch_size=32,
        learning_rate=1e-3,
        group_loss_weight=0.3,
        architecture="efficientnet_v2_s",   # 84.2% ImageNet top-1 vs ResNet50 80.9%
        transform_mode="standard",
        data_dir="../../dades",
        img_size=300,
    )
    model = model_pipeline(config)

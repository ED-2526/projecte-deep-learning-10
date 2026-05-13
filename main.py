import random
import wandb
import numpy as np
import torch

from train import train
from test import test
from utils.utils import make
from models.models import unfreeze_top_layers

torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _log_training_charts(history):
    train_losses = [m.get("Loss/P1_train", m.get("Loss/P2_train")) for m in history]
    val_losses   = [m.get("Loss/P1_val",   m.get("Loss/P2_val"))   for m in history]
    train_accs   = [m.get("Acc/P1_train",  m.get("Acc/P2_train"))  for m in history]
    val_accs     = [m.get("Acc/P1_val",    m.get("Acc/P2_val"))    for m in history]
    epochs = list(range(len(history)))
    wandb.log({
        "charts/Loss": wandb.plot.line_series(
            xs=[epochs, epochs], ys=[train_losses, val_losses],
            keys=["train", "val"], title="Loss", xname="epoch",
        ),
        "charts/Acc": wandb.plot.line_series(
            xs=[epochs, epochs], ys=[train_accs, val_accs],
            keys=["train", "val"], title="Accuracy", xname="epoch",
        ),
    })


def model_pipeline(cfg: dict, run_name: str = None) -> None:
    with wandb.init(project="projecte-final", name=run_name, config=cfg):
        config = wandb.config

        (model, train_loader, val_loader,
         criterion, group_criterion,
         optimizer, encoder, val_df) = make(config, device=device)

        wandb.watch(model, log="gradients", log_freq=100)

        history = []
        train(model, train_loader, val_loader, criterion, group_criterion, optimizer,
              config, device=device, phase=1, history=history)

        if config.finetune:
            unfreeze_top_layers(model, num_blocks=config.num_unfreeze_blocks)
            optimizer_ft = torch.optim.AdamW([
                {"params": model.features[-config.num_unfreeze_blocks:].parameters(),
                 "lr": config.learning_rate * config.finetune_lr_factor},
                {"params": model.fc.parameters(),         "lr": config.learning_rate},
                {"params": model.group_head.parameters(), "lr": config.learning_rate},
            ], weight_decay=config.weight_decay)

            train(model, train_loader, val_loader, criterion, group_criterion, optimizer_ft,
                  config, device=device, epochs=config.finetune_epochs, phase=2, history=history)

        _log_training_charts(history)
        test(model, val_df, config, device=device, class_names=list(encoder.classes_))
        return model


if __name__ == "__main__":
    wandb.login()

    config = dict(
        # ── Entrenament ──────────────────────────────────────────
        epochs=30,              # màxim d'èpoques a la fase 1
        patience=10,            # èpoques sense millora abans d'early stop / reduir LR
        finetune=True,          # True = executa fase 2 (fine-tuning backbone)
        finetune_epochs=15,     # màxim d'èpoques a la fase 2
        batch_size=64,
        learning_rate=5e-4,

        # ── Dades ────────────────────────────────────────────────
        val_split=0.2,          # fracció del dataset per a validació (0.2 = 20%)
        data_dir="../../dades",
        img_size=300,
        pretrained=True,            # False = pesos aleatoris (sense ImageNet)
        transform_mode="pad",       # "pad" = padding negre per mantenir aspect ratio

        # ── Pèrdua ───────────────────────────────────────────────
        label_smoothing_eps=0.2,    # suavitzat d'etiquetes (0 = desactivat)
        breed_loss_weight=1.0,      # pes de la pèrdua de races
        group_loss_weight=0.7,      # pes de la pèrdua de grups (0 = entrena només races)

        # ── Regularització ───────────────────────────────────────
        dropout=0.1,
        weight_decay=0,

        # ── Fine-tuning ──────────────────────────────────────────
        finetune_lr_factor=0.1,     # LR del backbone = learning_rate × finetune_lr_factor
        num_unfreeze_blocks=4,      # blocs finals del backbone a descongelar
        scheduler_factor=0.3,       # factor de reducció del LR en plateau
    )

    run_name = input("Nom del run (Enter per nom automàtic): ").strip() or None
    model_pipeline(config, run_name=run_name)

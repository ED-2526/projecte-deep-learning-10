import copy
import wandb
import torch


def train(model, train_loader, val_loader, criterion, group_criterion, optimizer,
<<<<<<< HEAD
          config, device="cuda", epochs=None, phase=1, history=None):

    patience = config.patience
    n_epochs = epochs if epochs is not None else config.epochs

=======
          config, device="cuda", epochs=None, phase=1):
    """
    Train for one phase (frozen backbone or fine-tuning) with early stopping.

    Each epoch runs a full training pass and a validation pass.  The best weights
    (lowest val_loss) are saved in memory and restored at the end, so the returned
    model is always the best checkpoint seen during this phase.

    Parameters
    ----------
    criterion       : loss for the main breed head (120 classes)
    group_criterion : loss for the auxiliary group head (16 groups)
    phase           : 1 = head-only training, 2 = fine-tuning last conv block
    """
    patience  = config.patience
    n_epochs  = epochs if epochs is not None else config.epochs

    # ReduceLROnPlateau halves the LR whenever val_loss has not improved for
    # `patience` consecutive epochs — prevents oscillation around a minimum.
>>>>>>> 86c1a423608215f3477f50674bb6022c321b0cf7
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience, factor=config.scheduler_factor
    )

    best_val_loss     = float("inf")
    best_weights      = copy.deepcopy(model.state_dict())  # snapshot of initial weights
    epochs_no_improve = 0

    for epoch in range(n_epochs):
<<<<<<< HEAD
=======
        # ---- training pass (gradients ON) ----
>>>>>>> 86c1a423608215f3477f50674bb6022c321b0cf7
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, group_criterion,
            optimizer, device, config, training=True
        )
<<<<<<< HEAD
=======
        # ---- validation pass (gradients OFF, no parameter update) ----
>>>>>>> 86c1a423608215f3477f50674bb6022c321b0cf7
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, group_criterion,
            None, device, config, training=False
        )

        # Adjust LR based on validation loss plateau
        scheduler.step(val_loss)

        metrics = {
            f"Loss/P{phase}_train": train_loss,
            f"Loss/P{phase}_val":   val_loss,
            f"Acc/P{phase}_train":  train_acc,
            f"Acc/P{phase}_val":    val_acc,
            f"LR/P{phase}":         optimizer.param_groups[0]["lr"],
        }
        wandb.log(metrics)
        if history is not None:
            history.append(metrics)

        print(
            f"[Phase {phase}] Epoch {epoch+1}/{n_epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
        )

        # Early stopping: keep the best weights, stop if no improvement for `patience` epochs
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            best_weights      = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Phase {phase}] Early stopping activat.")
                break

    # Restore the checkpoint with the lowest validation loss
    model.load_state_dict(best_weights)
    print(f"[Phase {phase}] Millors pesos restaurats (val_loss: {best_val_loss:.4f})")


def _run_epoch(model, loader, criterion, group_criterion, optimizer, device, config, training=True):
<<<<<<< HEAD
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    breed_w = getattr(config, "breed_loss_weight", 1.0)
    group_w = getattr(config, "group_loss_weight", 0.7)

    with torch.set_grad_enabled(training):
        for images, breed_labels, group_labels in loader:
            images       = images.to(device)
            breed_labels = breed_labels.to(device)
            group_labels = group_labels.to(device)

            breed_out, group_out = model(images)

            loss = breed_w * criterion(breed_out, breed_labels) \
                 + group_w * group_criterion(group_out, group_labels)
=======
    """
    Run one full pass over `loader`.

    Design rationale:
    - TRAINING pass  → computes loss + accuracy.  Loss drives backprop.
    - VALIDATION pass → computes loss + accuracy for per-epoch monitoring.
                        This is a LIGHTWEIGHT pass (standard 1-crop transform).
                        It is NOT redundant with test(): test() uses TTA × 10 crops
                        and runs only once at the end for the authoritative result.
                        Here we need val_loss every epoch for the LR scheduler and
                        early stopping, so the forward pass is mandatory anyway —
                        computing accuracy from the already-computed logits adds
                        negligible cost (just an argmax).

    Combined loss:
        loss = breed_loss  +  group_loss_weight × group_loss

    Returns
    -------
    avg_loss : float  — mean combined loss per batch
    accuracy : float  — top-1 breed accuracy (train or val)
    """
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    # Weight that scales the auxiliary group loss relative to the main breed loss.
    # 0.3 means group supervision contributes 23% of the total gradient signal.
    group_w = getattr(config, "group_loss_weight", 0.3)

    # torch.set_grad_enabled avoids allocating gradient tensors during validation,
    # which saves memory and speeds up the pass.
    with torch.set_grad_enabled(training):
        for images, breed_labels, group_labels in loader:
            # Move tensors to GPU (or CPU if no GPU available)
            images       = images.to(device)
            breed_labels = breed_labels.to(device)   # shape: (B,)
            group_labels = group_labels.to(device)   # shape: (B,)

            # Forward pass: model returns logits from both heads
            breed_out, group_out = model(images)
            # breed_out shape: (B, 120)  — raw scores before softmax
            # group_out shape: (B, num_groups)

            # CrossEntropyLoss with label smoothing: combines log-softmax + NLLLoss
            # and distributes ε probability mass uniformly across wrong classes.
            breed_loss = criterion(breed_out, breed_labels)
            group_loss = group_criterion(group_out, group_labels)

            # Hierarchical combined loss: main task + weighted auxiliary task
            loss = breed_loss + group_w * group_loss
>>>>>>> 86c1a423608215f3477f50674bb6022c321b0cf7

            if training:
                optimizer.zero_grad()   # clear gradients from previous batch
                loss.backward()         # backpropagate through both heads
                optimizer.step()        # update only requires_grad=True parameters

            total_loss += loss.item()
<<<<<<< HEAD
=======

            # Top-1 prediction: argmax over logits (the forward pass is already done,
            # so this adds no extra compute — just a comparison over 120 values per sample)
>>>>>>> 86c1a423608215f3477f50674bb6022c321b0cf7
            _, predicted = torch.max(breed_out, 1)
            correct += (predicted == breed_labels).sum().item()
            total   += breed_labels.size(0)

    return total_loss / len(loader), correct / total

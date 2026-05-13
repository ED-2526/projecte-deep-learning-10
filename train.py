import copy
import wandb
import torch


def train(model, train_loader, val_loader, criterion, group_criterion, optimizer,
          config, device="cuda", epochs=None, phase=1, history=None):

    patience = config.patience
    n_epochs = epochs if epochs is not None else config.epochs

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience, factor=config.scheduler_factor
    )

    best_val_loss     = float("inf")
    best_weights      = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, group_criterion,
            optimizer, device, config, training=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, group_criterion,
            None, device, config, training=False
        )

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

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            best_weights      = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Phase {phase}] Early stopping activat.")
                break

    model.load_state_dict(best_weights)
    print(f"[Phase {phase}] Millors pesos restaurats (val_loss: {best_val_loss:.4f})")


def _run_epoch(model, loader, criterion, group_criterion, optimizer, device, config, training=True):
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

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(breed_out, 1)
            correct += (predicted == breed_labels).sum().item()
            total   += breed_labels.size(0)

    return total_loss / len(loader), correct / total

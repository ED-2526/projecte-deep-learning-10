import copy
import wandb
import torch


def train(model, train_loader, val_loader, criterion, optimizer, config, device="cuda", epochs=None, phase=1):
    patience  = config.patience
    n_epochs  = epochs if epochs is not None else config.epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience, factor=0.5
    )

    best_val_loss     = float("inf")
    best_weights      = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss,   val_acc   = _run_epoch(model, val_loader,   criterion, None,      device, training=False)

        scheduler.step(val_loss)

        wandb.log({
            f"phase{phase}/epoch":      epoch + 1,
            f"phase{phase}/train_loss": train_loss,
            f"phase{phase}/train_acc":  train_acc,
            f"phase{phase}/val_loss":   val_loss,
            f"phase{phase}/val_acc":    val_acc,
            f"phase{phase}/lr":         optimizer.param_groups[0]["lr"],
        })

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
                print(f"[Phase {phase}] Early stopping: val_loss no ha millorat en {patience} èpoques.")
                break

    model.load_state_dict(best_weights)
    print(f"[Phase {phase}] Millors pesos restaurats (val_loss: {best_val_loss:.4f})")


def _run_epoch(model, loader, criterion, optimizer, device, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total

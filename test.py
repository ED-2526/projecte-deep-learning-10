import matplotlib
<<<<<<< HEAD
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
=======
matplotlib.use("Agg")   # must be set before any other matplotlib import (headless SSH server)

import numpy as np
import matplotlib.pyplot as plt
>>>>>>> 86c1a423608215f3477f50674bb6022c321b0cf7
import wandb
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
<<<<<<< HEAD
from PIL import Image
=======
>>>>>>> 86c1a423608215f3477f50674bb6022c321b0cf7

from utils.utils import DogBreedDataset, BREED_TO_GROUP


def test(model, val_df, config, device="cuda", class_names=None):
<<<<<<< HEAD
    img_size   = getattr(config, "img_size",   224)
    batch_size = getattr(config, "batch_size",  32)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)

=======
    """
    Final evaluation with Test-Time Augmentation (TTA).

    TTA runs inference on 10 augmented views of each image (TenCrop = 4 corners +
    centre, each with and without horizontal flip), averages the breed logits, then
    takes the argmax.  Averaging over multiple views reduces the variance that comes
    from a single crop choice and typically adds ~1-2% top-1 accuracy over a plain
    centre-crop evaluation.

    This function replaces the naive val_loader re-pass: unlike the training val loop
    (which uses standard transforms for speed), here we spend the extra compute on TTA
    and rich diagnostics (confusion matrix, per-class breakdown, group accuracy) that
    are only needed once at the very end.

    Logs to wandb
    -------------
    test/top1_acc         — top-1 accuracy with TTA
    test/top5_acc         — top-5 accuracy with TTA
    test/group_acc        — fraction of samples where the predicted breed belongs to
                            the correct visual group (measures coarse correctness)
    test/confusion_matrix — 120×120 row-normalised heatmap (static PNG, fast to load)
    test/top20_confusions — wandb.Table of the 20 most common misclassifications
    """
    img_size   = getattr(config, "img_size",   224)
    batch_size = getattr(config, "batch_size",  32)

    mean      = [0.485, 0.456, 0.406]
    std       = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)

    # TenCrop: returns a tuple of 10 PIL images (4 corners + centre, ×2 with H-flip).
    # The Lambda then converts each crop to a normalised tensor and stacks them,
    # producing a (10, C, H, W) tensor per image.
>>>>>>> 86c1a423608215f3477f50674bb6022c321b0cf7
    tta_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.TenCrop(img_size),
        transforms.Lambda(
            lambda crops: torch.stack([
<<<<<<< HEAD
                transforms.Compose([transforms.ToTensor(), normalize])(c) for c in crops
=======
                transforms.Compose([transforms.ToTensor(), normalize])(c)
                for c in crops
>>>>>>> 86c1a423608215f3477f50674bb6022c321b0cf7
            ])
        ),
    ])

<<<<<<< HEAD
    tta_loader = DataLoader(
        DogBreedDataset(val_df, transform=tta_transform),
        batch_size=max(1, batch_size // 2),
        shuffle=False, pin_memory=True, num_workers=2,
    )

    model.eval()
    num_classes  = len(class_names) if class_names else 120
    all_preds, all_labels = [], []
=======
    tta_dataset = DogBreedDataset(val_df, transform=tta_transform)
    # Use smaller batches: each image expands to 10 crops inside the model
    tta_loader  = DataLoader(
        tta_dataset,
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    model.eval()
    num_classes = len(class_names) if class_names else 120

    all_preds  = []
    all_labels = []
>>>>>>> 86c1a423608215f3477f50674bb6022c321b0cf7
    correct_top5 = 0
    total        = 0

    with torch.no_grad():
        for images, breed_labels, _group_labels in tta_loader:
<<<<<<< HEAD
            bs, ncrops, c, h, w = images.size()
            images_flat  = images.view(bs * ncrops, c, h, w).to(device)
            breed_labels = breed_labels.to(device)

            breed_out, _ = model(images_flat)
            breed_out = breed_out.view(bs, ncrops, -1).mean(dim=1)

            _, predicted = torch.max(breed_out, 1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(breed_labels.cpu().numpy())

            _, top5_idx = breed_out.topk(5, dim=1)
            correct_top5 += top5_idx.eq(breed_labels.view(-1, 1).expand_as(top5_idx)).sum().item()
            total += bs

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    top1 = (all_preds == all_labels).sum() / total
    top5 = correct_top5 / total

    print(f"\nTest Top-1 Accuracy (TTA×10): {top1*100:.2f}%")
    print(f"Test Top-5 Accuracy (TTA×10): {top5*100:.2f}%")

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1
=======
            # images shape: (B, 10, C, H, W)  — 10 augmented views per image
            bs, ncrops, c, h, w = images.size()

            # Flatten crops into the batch dimension so a single forward pass handles all
            images_flat = images.view(bs * ncrops, c, h, w).to(device)
            breed_labels = breed_labels.to(device)

            # Forward through the breed head only (group_out not needed at test time)
            breed_out, _ = model(images_flat)   # (B*10, num_classes)

            # Average logits over the 10 crops → (B, num_classes)
            # Averaging logits before softmax is equivalent to a geometric mean of
            # probabilities and is standard practice for TTA.
            breed_out = breed_out.view(bs, ncrops, -1).mean(dim=1)

            # Top-1 prediction
            _, predicted = torch.max(breed_out, 1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(breed_labels.cpu().numpy())

            # Top-5: check if true label is in the 5 highest-scoring classes
            _, top5_idx = breed_out.topk(5, dim=1)
            correct_top5 += top5_idx.eq(breed_labels.view(-1, 1).expand_as(top5_idx)).sum().item()
            total += bs

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    top1 = (all_preds == all_labels).sum() / total
    top5 = correct_top5 / total

    print(f"\nTest Top-1 Accuracy (TTA × 10): {top1*100:.2f}%")
    print(f"Test Top-5 Accuracy (TTA × 10): {top5*100:.2f}%")

    # ------------------------------------------------------------------ #
    #  Confusion matrix (120 × 120, row-normalised)
    # ------------------------------------------------------------------ #
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    # Normalise each row by the number of true samples in that class.
    # A perfect classifier has 1.0 on the diagonal and 0.0 everywhere else.
>>>>>>> 86c1a423608215f3477f50674bb6022c321b0cf7
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if class_names:
<<<<<<< HEAD
        ax.set_xticks(range(num_classes)); ax.set_xticklabels(class_names, rotation=90, fontsize=4)
        ax.set_yticks(range(num_classes)); ax.set_yticklabels(class_names, fontsize=4)
    ax.set_xlabel("Predicted breed"); ax.set_ylabel("True breed")
    ax.set_title("Confusion matrix — row-normalised")
    plt.tight_layout()

    confused_pairs = sorted(
        [(class_names[i], class_names[j], int(cm[i, j]))
         for i in range(num_classes) for j in range(num_classes)
         if i != j and cm[i, j] > 0],
        key=lambda x: -x[2]
    )

    per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    if class_names:
        ranked = sorted(zip(class_names, per_class_acc), key=lambda x: x[1])
        print("\nPitjors 10 races:"); [print(f"  {n:45s} {a*100:5.1f}%") for n, a in ranked[:10]]
        print("\nMillors 10 races:"); [print(f"  {n:45s} {a*100:5.1f}%") for n, a in ranked[-10:]]

    group_acc = _group_accuracy(all_preds, all_labels, class_names)
    print(f"\nAccuracy de grup: {group_acc*100:.2f}%")

    confused_images = _confused_images(all_preds, all_labels, val_df, confused_pairs, class_names)
    gradcam_images  = _confused_gradcam_images(
        model, all_preds, all_labels, val_df, confused_pairs, class_names, config, device
    )

    wandb.log({
        "test/top1_acc":          top1,
        "test/top5_acc":          top5,
        "test/group_acc":         group_acc,
        "test/confusion_matrix":  wandb.Image(fig),
        "test/top20_confusions":  wandb.Table(columns=["True", "Predicted", "Count"],
                                               data=confused_pairs[:20]),
        "test/confused_examples": confused_images,
        "test/gradcam_confused":  gradcam_images,
    })
    plt.close(fig)
    return top1


def _confused_images(all_preds, all_labels, val_df, confused_pairs, class_names, n_pairs=5):
    name_to_idx  = {name: i for i, name in enumerate(class_names)}
    val_df_reset = val_df.reset_index(drop=True)
    images = []
    for true_name, pred_name, _ in confused_pairs[:n_pairs]:
        true_idx = name_to_idx.get(true_name)
        pred_idx = name_to_idx.get(pred_name)
        if true_idx is None or pred_idx is None:
            continue
        real_idxs = np.where(all_labels == true_idx)[0]
        if len(real_idxs) > 0:
            img = Image.open(val_df_reset.loc[int(real_idxs[0]), "image_path"]).convert("RGB").resize((300, 300))
            images.append(wandb.Image(img, caption=f"Real: {true_name}"))
        pred_idxs = np.where(all_labels == pred_idx)[0]
        if len(pred_idxs) > 0:
            img = Image.open(val_df_reset.loc[int(pred_idxs[0]), "image_path"]).convert("RGB").resize((300, 300))
            images.append(wandb.Image(img, caption=f"Predita: {pred_name}"))
    return images


def _group_accuracy(all_preds, all_labels, class_names):
    if class_names is None or len(all_preds) == 0:
        return 0.0
    idx_to_group = {i: BREED_TO_GROUP.get(name, f"__unknown_{i}__") for i, name in enumerate(class_names)}
    hits = sum(1 for p, t in zip(all_preds, all_labels)
               if idx_to_group.get(int(p), "?") == idx_to_group.get(int(t), "?"))
    return hits / len(all_preds)


# ──────────────────────────── GradCAM ────────────────────────────

def _compute_grad_cam(model, img_tensor, target_class):
    activations, gradients = [], []
    target_layer = model.features[-1]
    h_fwd = target_layer.register_forward_hook(lambda _m, _i, o: activations.append(o))
    h_bwd = target_layer.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0]))

    model.zero_grad()
    breed_out, _ = model(img_tensor)
    breed_out[0, target_class].backward()

    h_fwd.remove(); h_bwd.remove()
    acts    = activations[0]
    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam     = torch.relu((weights * acts).sum(dim=1)).squeeze().cpu().detach().float().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)
    return cam


def _overlay_heatmap(img_pil, cam, display_size=300):
    img_arr  = np.array(img_pil.resize((display_size, display_size))).astype(np.float32) / 255.0
    cam_arr  = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (display_size, display_size), Image.BILINEAR)).astype(np.float32) / 255.0
    heatmap  = mpl_cm.get_cmap("jet")(cam_arr)[:, :, :3]
    blended  = np.clip((0.5 * img_arr + 0.5 * heatmap) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def _confused_gradcam_images(model, all_preds, all_labels, val_df,
                              confused_pairs, class_names, config, device, n_pairs=5):
    if class_names is None:
        return []

    img_size   = getattr(config, "img_size", 224)
    mean, std  = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    name_to_idx  = {name: i for i, name in enumerate(class_names)}
    val_df_reset = val_df.reset_index(drop=True)
    wandb_images = []

    for true_name, pred_name, _ in confused_pairs[:n_pairs]:
        true_idx = name_to_idx.get(true_name)
        pred_idx = name_to_idx.get(pred_name)
        if true_idx is None or pred_idx is None:
            continue
        error_idxs = np.where((all_labels == true_idx) & (all_preds == pred_idx))[0]
        if len(error_idxs) == 0:
            continue
        img_pil    = Image.open(val_df_reset.loc[int(error_idxs[0]), "image_path"]).convert("RGB")
        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

        cam_pred = _compute_grad_cam(model, img_tensor, pred_idx)
        wandb_images.append(wandb.Image(_overlay_heatmap(img_pil, cam_pred),
                                        caption=f"Predita: {pred_name}  (real: {true_name})"))
        cam_true = _compute_grad_cam(model, img_tensor, true_idx)
        wandb_images.append(wandb.Image(_overlay_heatmap(img_pil, cam_true),
                                        caption=f"Real: {true_name}  (activació correcta)"))
    return wandb_images
=======
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=90, fontsize=4)
        ax.set_yticklabels(class_names, fontsize=4)
    ax.set_xlabel("Predicted breed", fontsize=10)
    ax.set_ylabel("True breed",      fontsize=10)
    ax.set_title("Confusion matrix — row-normalised (diagonal = recall per class)", fontsize=11)
    plt.tight_layout()

    # ------------------------------------------------------------------ #
    #  Top-20 most confused pairs
    # ------------------------------------------------------------------ #
    confused_pairs = [
        (
            class_names[i] if class_names else str(i),
            class_names[j] if class_names else str(j),
            int(cm[i, j]),
        )
        for i in range(num_classes)
        for j in range(num_classes)
        if i != j and cm[i, j] > 0
    ]
    confused_pairs.sort(key=lambda x: -x[2])
    top_confused_table = wandb.Table(
        columns=["True", "Predicted", "Count"],
        data=confused_pairs[:20],
    )

    # ------------------------------------------------------------------ #
    #  Per-class accuracy
    # ------------------------------------------------------------------ #
    per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    if class_names:
        class_acc_pairs = sorted(zip(class_names, per_class_acc), key=lambda x: x[1])
        print("\nWorst 10 breeds (hardest to recognise):")
        for name, acc in class_acc_pairs[:10]:
            print(f"  {name:45s} {acc*100:5.1f}%")
        print("\nBest 10 breeds (easiest to recognise):")
        for name, acc in class_acc_pairs[-10:]:
            print(f"  {name:45s} {acc*100:5.1f}%")

    # ------------------------------------------------------------------ #
    #  Group-level accuracy
    # ------------------------------------------------------------------ #
    # "Group hit": the predicted breed belongs to the same visual group as the
    # true breed (even if the specific breed is wrong).  This metric tells us
    # whether the model at least gets the *type* of dog right.
    group_acc = _group_accuracy(all_preds, all_labels, class_names)
    print(f"\nGroup-level accuracy (correct visual group): {group_acc*100:.2f}%")

    # ------------------------------------------------------------------ #
    #  Log everything to wandb
    # ------------------------------------------------------------------ #
    wandb.log({
        "test/top1_acc":         top1,
        "test/top5_acc":         top5,
        "test/group_acc":        group_acc,
        "test/confusion_matrix": wandb.Image(fig),    # static PNG, loads instantly
        "test/top20_confusions": top_confused_table,
    })
    plt.close(fig)   # free memory: matplotlib figures are not garbage-collected
    return top1


def _group_accuracy(all_preds, all_labels, class_names):
    """
    Fraction of samples where the predicted breed is in the same visual group
    as the true breed.  Unmapped breeds are assigned a unique sentinel so they
    never produce a false group-hit.
    """
    if class_names is None or len(all_preds) == 0:
        return 0.0

    # Map each class index to its group name; unmapped breeds get a unique sentinel
    idx_to_group = {
        i: BREED_TO_GROUP.get(name, f"__unknown_{i}__")
        for i, name in enumerate(class_names)
    }

    hits = sum(
        1 for pred, true in zip(all_preds, all_labels)
        if idx_to_group.get(int(pred), "?") == idx_to_group.get(int(true), "?")
    )
    return hits / len(all_preds)
>>>>>>> 86c1a423608215f3477f50674bb6022c321b0cf7

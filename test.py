import wandb
import torch


def test(model, loader, device="cuda"):
    model.eval()
    correct_top1, correct_top5, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            correct_top1 += (predicted == labels).sum().item()

            _, top5 = outputs.topk(5, dim=1)
            correct_top5 += top5.eq(labels.view(-1, 1).expand_as(top5)).sum().item()

            total += labels.size(0)

    top1 = correct_top1 / total
    top5 = correct_top5 / total

    print(f"Test Top-1 Accuracy: {top1*100:.2f}%")
    print(f"Test Top-5 Accuracy: {top5*100:.2f}%")

    wandb.log({"test_top1_acc": top1, "test_top5_acc": top5})
    return top1

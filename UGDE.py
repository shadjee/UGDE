# ===== IMPORTS =====
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

# ===== DEVICE =====
device = torch.device('mps' if torch.backends.mps.is_available() else
                      'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ===== PARAMETERS =====
data_dir = '/BMC_processed'
batch_size = 128
num_classes = 21
learning_rate = 0.001
num_epochs = 15
mc_samples = 20

# ===== TRANSFORMS =====
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ===== DATASET & LOADER =====
dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
class_names = dataset.classes

# ===== MODELS =====
model_mobilenet_v3 = models.mobilenet_v3_large(pretrained=True)
model_mobilenet_v3.classifier[3] = nn.Linear(model_mobilenet_v3.classifier[3].in_features, num_classes)

model_efficientnet_b0 = EfficientNet.from_pretrained('efficientnet-b0')
model_efficientnet_b0._fc = nn.Linear(model_efficientnet_b0._fc.in_features, num_classes)

model_shufflenet_v2 = models.shufflenet_v2_x1_0(pretrained=True)
model_shufflenet_v2.fc = nn.Linear(model_shufflenet_v2.fc.in_features, num_classes)

models_list = [model_mobilenet_v3.to(device), model_efficientnet_b0.to(device), model_shufflenet_v2.to(device)]

def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
for model in models_list:
    enable_mc_dropout(model)

criterion = nn.CrossEntropyLoss()
optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in models_list]
schedulers = [optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5) for opt in optimizers]

import time
from tqdm import tqdm

def train_model_with_timing(model, optimizer, scheduler, num_epochs, model_name="model"):
    start_time = time.time()
    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct_preds, total_preds = 0, 0, 0

        loop = tqdm(train_loader, desc=f'{model_name} | Epoch {epoch+1}/{num_epochs}', leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=correct_preds/total_preds)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_preds / total_preds
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # Validation
        model.eval()
        val_running_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        print(f"{model_name} | Epoch {epoch+1}/{num_epochs} — Train Loss: {epoch_train_loss:.4f}, "
              f"Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        scheduler.step()

    total_time = time.time() - start_time
    print(f"\n⏱️ Total training time for {model_name}: {total_time:.2f} seconds\n")
    return train_loss, train_acc, val_loss, val_acc, total_time



# ===== TRAINING FUNCTION =====
def train_model(model, optimizer, scheduler, num_epochs):
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct_preds, total_preds = 0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)
        train_loss.append(running_loss / len(train_loader.dataset))
        train_acc.append(correct_preds / total_preds)

        model.eval()
        val_running_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss.append(val_running_loss / len(val_loader.dataset))
        val_acc.append(val_correct / val_total)
        scheduler.step()
    return train_loss, train_acc, val_loss, val_acc

def plot_training_curves(metrics_list, num_epochs):
    for i, metrics in enumerate(metrics_list):
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, metrics['train_loss'], label='Train Loss')
        plt.plot(epochs, metrics['val_loss'], label='Val Loss')
        plt.title(f'Model {i+1} Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, metrics['train_acc'], label='Train Acc')
        plt.plot(epochs, metrics['val_acc'], label='Val Acc')
        plt.title(f'Model {i+1} Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()

# ===== ENSEMBLE FUNCTIONS =====
def mc_dropout_predictions(models, inputs, n_samples=20):
    all_probs = []
    for _ in range(n_samples):
        probs_list = []
        for model in models:
            outputs = F.softmax(model(inputs), dim=1)
            probs_list.append(outputs)
        avg_probs = torch.mean(torch.stack(probs_list), dim=0)
        all_probs.append(avg_probs)
    all_probs = torch.stack(all_probs)
    mean_probs = torch.mean(all_probs, dim=0)
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-6), dim=1)
    return mean_probs, entropy

def uncertainty_guided_attention_ensemble(models, inputs, n_samples=20):
    probs, entropy = mc_dropout_predictions(models, inputs, n_samples)
    _, preds = torch.max(probs, 1)
    return preds, probs, entropy

def plot_confusion_matrix(true_labels, pred_labels, class_names, title='Confusion Matrix'):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title(title)
    plt.tight_layout(); plt.show()

def plot_calibration_curve(y_true, y_probs, title='Calibration Curve'):
    y_pred = np.argmax(y_probs, axis=1)
    y_prob_max = np.max(y_probs, axis=1)
    correct = (np.array(y_true) == y_pred).astype(int)
    prob_true, prob_pred = calibration_curve(correct, y_prob_max, n_bins=10)
    plt.figure(figsize=(8,6))
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1],[0,1],'--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Fraction of Correct Predictions')
    plt.title(title); plt.grid(); plt.legend(); plt.show()


def visualize_gradcam_uncertainty(model, models, dataset, class_names, num_classes=9):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    model.eval()
    cam = GradCAM(model=model, target_layers=[model.features[-1]])
    plt.figure(figsize=(12, 4 * num_classes))

    shown_classes = set()
    for img_idx, (img, label) in enumerate(dataset):
        if label in shown_classes:
            continue
        shown_classes.add(label)

        # Prepare input
        input_tensor = img.unsqueeze(0).to(device)

        # Get prediction and entropy
        with torch.no_grad():
            class_idx = torch.argmax(model(input_tensor)).item()
            _, _, entropy_map = uncertainty_guided_attention_ensemble(models, input_tensor, n_samples=mc_samples)
        entropy_val = entropy_map.detach().cpu().numpy()[0]

        # Generate Grad-CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)])
        grayscale_cam = grayscale_cam[0, :]  # remove batch dimension

        # Normalize input image for overlay
        input_image = input_tensor[0].cpu().permute(1, 2, 0).numpy()
        input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())

        # Generate CAM overlay
        cam_img = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)

        row = label
        plt.subplot(num_classes, 3, row * 3 + 1)
        plt.imshow(input_image)
        plt.title(f'Original - {class_names[label]}')
        plt.axis('off')

        plt.subplot(num_classes, 3, row * 3 + 2)
        plt.imshow(cam_img)
        plt.title(f'Grad-CAM\nEntropy: {entropy_val:.3f}')
        plt.axis('off')

        plt.subplot(num_classes, 3, row * 3 + 3)
        plt.text(0.1, 0.5, f'Predicted: {class_names[class_idx]}\nEntropy: {entropy_val:.3f}', fontsize=12)
        plt.axis('off')

        if len(shown_classes) == num_classes:
            break

    plt.tight_layout()
    plt.show()


# ===== MAIN =====

if __name__ == "__main__":
    all_metrics = []
    for idx, (model, optimizer, scheduler) in enumerate(zip(models_list, optimizers, schedulers)):
        print(f"\nTraining Model {idx+1}")
        train_loss, train_acc, val_loss, val_acc, training_time = train_model_with_timing(
            model, optimizer, scheduler, num_epochs, model_name=f"Model_{idx+1}"
        )

        all_metrics.append({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        # Save model to disk
        torch.save(model.state_dict(), f'model_{idx+1}.pth')

        # Plot curves
        plot_training_curves([all_metrics[idx]], num_epochs)

        # Inference time
        inference_time = evaluate_model_inference_time(model, val_loader, model_name=f"Model_{idx+1}")

        # Evaluation
        all_labels, all_preds = [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        print(f"\nClassification Report - Model {idx+1}")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        plot_confusion_matrix(all_labels, all_preds, class_names, title=f'Model {idx+1} Confusion Matrix')



    # ===== Ensemble Evaluation =====
    ensemble_preds, ensemble_labels, ensemble_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds, probs, _ = uncertainty_guided_attention_ensemble(models_list, inputs, n_samples=mc_samples)
            ensemble_preds.extend(preds.cpu().numpy())
            ensemble_labels.extend(labels.cpu().numpy())
            ensemble_probs.append(probs.cpu().numpy())
    ensemble_probs = np.concatenate(ensemble_probs, axis=0)

    print("\nClassification Report - UGAE Ensemble")
    print(classification_report(ensemble_labels, ensemble_preds, target_names=class_names))
    plot_confusion_matrix(ensemble_labels, ensemble_preds, class_names, title='UGAE Ensemble Confusion Matrix')
    plot_calibration_curve(ensemble_labels, ensemble_probs, title='UGAE Ensemble Calibration Curve')

    visualize_gradcam_uncertainty(models_list[0], models_list, val_dataset, class_names, num_classes=len(class_names))

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from timm import create_model
import numpy as np
from sklearn.metrics import matthews_corrcoef
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Constants
DATASET_PATH = "kvasir-dataset-v2"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 1
LR = 0.001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data Augmentation
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Dataset
dataset = ImageFolder(DATASET_PATH, transform=transform)
num_classes = len(dataset.classes)

# Dataset Split
indices = list(range(len(dataset)))
train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(test_idx, test_size=0.66, random_state=42)

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class HybridDenseNetViT(nn.Module):
    def __init__(self, num_classes):
        super(HybridDenseNetViT, self).__init__()

        self.densenet = create_model('densenet201', pretrained=True, num_classes=num_classes)
        self.vit = create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

        densenet_out_features = self.densenet.get_classifier().in_features
        vit_out_features = self.vit.head.in_features

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.vit_pool = nn.AdaptiveAvgPool1d(1)
        self.vit_fc = nn.Linear(vit_out_features, densenet_out_features)

        self.fc = nn.Sequential(
            nn.Linear(2 * densenet_out_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        densenet_features = self.densenet.forward_features(x)
        densenet_features = self.global_avg_pool(densenet_features)
        densenet_features = torch.flatten(densenet_features, start_dim=1)

        vit_features = self.vit.forward_features(x)
        vit_features = vit_features.permute(0, 2, 1)
        vit_features = self.vit_pool(vit_features).squeeze(-1)
        vit_features = self.vit_fc(vit_features)

        combined_features = torch.cat((densenet_features, vit_features), dim=1)
        return self.fc(combined_features)


# Initialize Model
DenseNetViT = HybridDenseNetViT(num_classes).to(DEVICE)


# Class-Balanced Loss
class ClassBalancedLoss(nn.Module):
    def __init__(self, beta, num_classes):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits, labels):
        with torch.no_grad():
            class_counts = torch.bincount(labels, minlength=self.num_classes).float()
            class_counts = torch.clamp(class_counts, min=1)
            effective_num = 1.0 - torch.pow(self.beta, class_counts)
            weights = (1.0 - self.beta) / (effective_num + 1e-8)
            weights = weights / weights.sum()

        return nn.CrossEntropyLoss(weight=weights.to(logits.device))(logits, labels)


loss_fn = ClassBalancedLoss(beta=0.999, num_classes=num_classes)
optimizer = optim.AdamW(DenseNetViT.parameters(), lr=LR, weight_decay=1e-2)


# Training Function
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


# Validation Function
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    preds_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).detach()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    mcc = matthews_corrcoef(labels_list, preds_list)
    return total_loss / len(loader), 100.0 * correct / total, mcc


# Model Conversion to ONNX & TFLite
def convert_to_tflite(model, device):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    onnx_filename = "densenet_vit_model.onnx"
    torch.onnx.export(model, dummy_input, onnx_filename, input_names=['input'], output_names=['output'],
                      opset_version=14)

    onnx_model = onnx.load(onnx_filename)
    tf_rep = prepare(onnx_model)
    tf_model_dir = "./tf_model_denseVit"
    tf_rep.export_graph(tf_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)

    def representative_dataset_gen():
        for images, _ in test_loader:
            images = images.to(device).numpy()
            yield [images]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen

    tflite_model = converter.convert()

    with open("densenet_vit_model.tflite", "wb") as f:
        f.write(tflite_model)


# Training Loop
best_mcc = 0
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train_loss, train_acc = train_epoch(DenseNetViT, train_loader, optimizer, loss_fn, DEVICE)
    val_loss, val_acc, val_mcc = validate_epoch(DenseNetViT, val_loader, loss_fn, DEVICE)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val MCC: {val_mcc:.4f}")

    if val_mcc > best_mcc:
        best_mcc = val_mcc
        torch.save(DenseNetViT.state_dict(), "best_densenet_vit_model.pth")
        print(f"Best model saved with MCC: {val_mcc:.4f}")
        convert_to_tflite(DenseNetViT, DEVICE)

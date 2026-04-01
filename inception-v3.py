import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import inception_v3
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd

# Veri kümesini yükleme
DATASET_PATH = "kvasir-dataset-v2"  # Buraya veri kümenizin yolunu ekleyin

# Görüntü boyutu ve batch size
target_size = (299, 299)  # Inception-v3 için uygun boyut
batch_size = 32

# Veri ön işleme
transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Tüm veri kümesini yükleme
full_dataset = ImageFolder(root=os.path.join(DATASET_PATH), transform=transform)

# Veri kümesini bölme (70% train, 20% validation, 10% test)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Inception-v3 Modeli
model = inception_v3(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8)  # 6 sınıf için güncellendi

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU 1 kullanımı için güncellendi
model.to(device)

print(f"using device: {device}")
devNumber=torch.cuda.current_device()
print(f"curennt device number is:{devNumber}")
devName=torch.cuda.get_device_name()
print(f"gpu name is {devName}")

# Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Sonuçları kaydetmek için boş bir DataFrame oluşturma
results = []

# Modeli eğitme
epochs = 1
train_losses, val_accuracies = [], []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # Doğrulama testi
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    val_accuracies.append(accuracy)

    # Sonuçları kaydet
    results.append([epoch + 1, train_losses[-1], accuracy, precision, recall, f1])
    print(
        f"Epoch {epoch + 1}/{epochs}, Loss: {train_losses[-1]:.4f}, Val Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Sonuçları Excel'e kaydetme
df_results = pd.DataFrame(results,
                          columns=["Epoch", "Train Loss", "Validation Accuracy", "Precision", "Recall", "F1 Score"])
df_results.to_excel("training_results_inception-v3.xlsx", index=False)

# Modeli test etme
test_results = []
test_correct, test_total = 0, 0
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
conf_matrix = confusion_matrix(all_labels, all_preds)

# Confusion matrix çizimi
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix_inception.png")
plt.show()

test_results.append([test_acc, precision, recall, f1])
df_test_results = pd.DataFrame(test_results, columns=["Test Accuracy", "Precision", "Recall", "F1 Score"])
df_test_results.to_excel("testing_results_inception-v3.xlsx", index=False)

print(f'Test Accuracy: {test_acc * 100:.2f}%')
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
print(f'Confusion Matrix:\n{conf_matrix}')

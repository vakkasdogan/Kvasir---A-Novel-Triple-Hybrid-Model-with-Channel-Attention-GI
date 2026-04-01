import torch
import torchvision
import torchvision.transforms as transforms
import timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt

# Veri kümesini yükleme
DATASET_PATH = "kvasir-dataset-v2"  # Buraya veri kümenizin yolunu ekleyin

# Görüntü boyutu ve batch size
target_size = (224, 224)
batch_size = 32

# Veri ön işleme
transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
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


# Vision Transformer Modeli
class ViTClassifier(nn.Module):
    def __init__(self, num_classes=8):  # 6 sınıf için güncellendi
        super(ViTClassifier, self).__init__()

        # Vision Transformer
        self.vit = timm.create_model("vit_large_patch16_224", pretrained=True)
        self.vit.head = nn.Linear(self.vit.num_features, num_classes)  # Son katmanı güncelle

    def forward(self, x):
        return self.vit(x)


# Modeli oluşturma
model = ViTClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU 1 kullanımı için güncellendi
model.to(device)

print(f"using device: {device}")
devNumber=torch.cuda.current_device()
print(f"curennt device number is:{devNumber}")
devName=torch.cuda.get_device_name()
print(f"gpu name is {devName}")

# Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Modeli eğitme
epochs = 15
train_losses, val_accuracies = [], []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # Doğrulama testi
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    val_accuracies.append(accuracy)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_losses[-1]:.4f}, Validation Accuracy: {accuracy:.4f}")

# Modelin doğruluğunu görselleştirme
plt.plot(range(epochs), val_accuracies, label='Doğrulama Doğruluğu')
plt.xlabel('Epok')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

# Modeli test etme
test_correct, test_total = 0, 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = test_correct / test_total
print(f'Test Doğruluk: {test_acc * 100:.2f}%')

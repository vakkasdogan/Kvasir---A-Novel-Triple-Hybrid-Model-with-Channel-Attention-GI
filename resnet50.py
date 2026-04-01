import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np
import pandas as pd
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from torchvision import datasets
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Veri kümesini yükleme
DATASET_PATH = "kvasir-dataset-v2"  # Buraya veri kümenizin yolunu ekleyin

# Görüntü boyutu ve batch size
target_size = (224, 224)  # ResNet50 için uygun boyut
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

# ResNet50 Modeli
model = resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8)  # 6 sınıf için güncellendi

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU 1 kullanımı için güncellendi
model.to(device)

# print(f"using device: {device}")
# devNumber=torch.cuda.current_device()
# print(f"curennt device number is:{devNumber}")
# devName=torch.cuda.get_device_name()
# print(f"gpu name is {devName}")

# Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Sonuçları kaydetmek için boş bir DataFrame oluşturma
results = []

# Modeli eğitme
epochs = 5
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

    # # Doğrulama testi
    # model.eval()
    # correct, total = 0, 0
    # all_preds, all_labels = [], []
    # with torch.no_grad():
    #     for images, labels in val_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #         all_preds.extend(predicted.cpu().numpy())
    #         all_labels.extend(labels.cpu().numpy())
    # accuracy = accuracy_score(all_labels, all_preds)
    # precision = precision_score(all_labels, all_preds, average='macro')
    # recall = recall_score(all_labels, all_preds, average='macro')
    # f1 = f1_score(all_labels, all_preds, average='macro')
    # val_accuracies.append(accuracy)

    # # Sonuçları kaydet
    # results.append([epoch + 1, train_losses[-1], accuracy, precision, recall, f1])
    # print(
    #     f"Epoch {epoch + 1}/{epochs}, Loss: {train_losses[-1]:.4f}, Val Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# # Sonuçları Excel'e kaydetme
# df_results = pd.DataFrame(results,
#                           columns=["Epoch", "Train Loss", "Validation Accuracy", "Precision", "Recall", "F1 Score"])
# df_results.to_excel("training_results_resnet50.xlsx", index=False)

# # Eğitim ve doğrulama kayıplarını görselleştirme
# plt.figure()
# plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
# plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Loss/Accuracy')
# plt.legend()
# plt.title('Training Loss & Validation Accuracy')
# plt.savefig("training_plot_resnet50.png")
# plt.show()

# # Test sonuçları
# test_results = []
# test_correct, test_total = 0, 0
# all_preds, all_labels = [], []
# all_probs = []
# model.eval()
# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         probs = torch.softmax(outputs, dim=1).cpu().numpy()
#         _, predicted = torch.max(outputs, 1)
#         all_probs.extend(probs)
#         all_preds.extend(predicted.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
#
# test_acc = accuracy_score(all_labels, all_preds)
# conf_matrix = confusion_matrix(all_labels, all_preds)

# # Confusion matrix çizimi
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=full_dataset.classes,
#             yticklabels=full_dataset.classes)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.savefig("confusion_matrix_resnet50.png")
# plt.show()
#
# # ROC-AUC Eğrisi
# all_labels_bin = label_binarize(all_labels, classes=range(8))
# all_probs = np.array(all_probs)
# fpr, tpr, _ = roc_curve(all_labels_bin.ravel(), all_probs.ravel())
# roc_auc = auc(fpr, tpr)
#
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc='lower right')
# plt.savefig("roc_curve_resnet50.png")
# plt.show()
#
# test_results.append([test_acc, precision, recall, f1])
# df_test_results = pd.DataFrame(test_results, columns=["Test Accuracy", "Precision", "Recall", "F1 Score"])
# df_test_results.to_excel("testing_results_resnet50.xlsx", index=False)
#
# print(f'Test Accuracy: {test_acc * 100:.2f}%')
# print(f'Confusion Matrix:\n{conf_matrix}')

# Step 5: Export the model to ONNX format
dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Example input
onnx_filename = "resnet50_model.onnx"
torch.onnx.export(model, dummy_input, onnx_filename, input_names=['input'], output_names=['output'], opset_version=11)

onnx_model = onnx.load(onnx_filename)
tf_rep = prepare(onnx_model)

tf_model_dir = "./tf_model"
tf_rep.export_graph(tf_model_dir)

# Step 8: Convert the TensorFlow model to TensorFlow Lite with optimization (quantization)
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)

# Apply post-training quantization to reduce model size
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Default quantization


# Convert the model
tflite_model = converter.convert()

# Step 9: Save the optimized TensorFlow Lite model
with open("resnet50_model_optimized.tflite", "wb") as f:
    f.write(tflite_model)

print("Optimized TensorFlow Lite model saved successfully!")


interpreter = tf.lite.Interpreter(model_path="resnet50_model_optimized.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Evaluate the TensorFlow Lite model
correct = 0
total = 0

# Loop through test data
for images, labels in test_loader:
    # Convert PyTorch tensor to numpy
    images_np = images.numpy()

    for i in range(len(images_np)):
        input_data = np.expand_dims(images_np[i], axis=0).astype(np.float32)  # Adjust shape for TFLite

        # Set the tensor to the input data
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Invoke the interpreter
        interpreter.invoke()

        # Get output predictions
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)

        # Compare prediction to the true label
        if prediction == labels[i].item():
            correct += 1
        total += 1

# Print the accuracy of the optimized TensorFlow Lite model
accuracy = correct / total
print(f'Optimized TensorFlow Lite Model Accuracy: {accuracy * 100:.2f}%')

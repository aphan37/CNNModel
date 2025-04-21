# Enhanced CNN Pipeline for Alzheimer's Detection
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# === Step 1: Data Preparation ===
commercial_file_path = 'commercial_nacc65a.csv'
labelled_image_folder = 'labeledNACCImages'
organized_dir = 'organizedNACCImages'
split_dir = 'dataset'

alzheimers_category = {
    0: 'NoAlzheimers',
    0.5: 'Mild',
    1: 'CognitivelyIntact',
    2: 'Moderate',
    3: 'Severe'
}

print("Loading classification data...")
classification_df = pd.read_csv(commercial_file_path)
classification_df['Type'] = classification_df['CDRGLOB'].map(alzheimers_category)

print("Organizing images...")
os.makedirs(organized_dir, exist_ok=True)
for label in alzheimers_category.values():
    os.makedirs(os.path.join(organized_dir, label), exist_ok=True)

for file_name in os.listdir(labelled_image_folder):
    if file_name.endswith('.jpg'):
        base_id = file_name.split('_')[0]
        match = classification_df.loc[classification_df['NACCID'].astype(str) == base_id]
        if not match.empty:
            classification_type = match['Type'].values[0]
            target_folder = os.path.join(organized_dir, classification_type)
            shutil.move(os.path.join(labelled_image_folder, file_name), os.path.join(target_folder, file_name))

print("Splitting dataset...")
split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
for split in split_ratios:
    for label in alzheimers_category.values():
        os.makedirs(os.path.join(split_dir, split, label), exist_ok=True)

for label in alzheimers_category.values():
    label_dir = os.path.join(organized_dir, label)
    images = [f for f in os.listdir(label_dir) if f.endswith('.jpg')]
    if images:
        train_files, temp_files = train_test_split(images, test_size=(1 - split_ratios['train']), random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=split_ratios['test'] / (split_ratios['val'] + split_ratios['test']), random_state=42)
        for split, split_files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
            for file_name in split_files:
                shutil.copy(os.path.join(label_dir, file_name), os.path.join(split_dir, split, label, file_name))

# === Step 2: AlzhiNet Model ===
class AlzhiNet(nn.Module):
    def __init__(self, num_classes):
        super(AlzhiNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 28 * 28, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# === Step 3: Training Loop ===
print("Training the model...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=os.path.join(split_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(split_dir, 'val'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AlzhiNet(num_classes=len(train_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_acc = 0.0

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Training Loss: {avg_loss:.4f}")

    # Validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print("Saved best model so far! ✅")

# === Step 4: Evaluation ===
print("Evaluating the model...")
test_dataset = datasets.ImageFolder(root=os.path.join(split_dir, 'test'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.load_state_dict(torch.load('best_model.pth'))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Final Test Accuracy: {100 * correct / total:.2f}%")

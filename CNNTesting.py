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

# Step 1: Load and organize data
# Define paths
commercial_file_path = 'commercial_nacc65a.csv'
labelled_image_folder = 'labeledNACCImages'
organized_dir = 'organizedNACCImages'
split_dir = 'dataset'

# Define classification categories
alzheimers_category = {
    0: 'NoAlzheimers',
    0.5: 'Mild',
    1: 'CognitivelyIntact',
    2: 'Moderate',
    3: 'Severe'
}

# Load the CSV file
print("Loading classification data...")
classification_df = pd.read_csv(commercial_file_path)
classification_df['Type'] = classification_df['CDRGLOB'].map(alzheimers_category)

# Organize images into subfolders
print("Organizing images...")
os.makedirs(organized_dir, exist_ok=True)
for label in alzheimers_category.values():
    os.makedirs(os.path.join(organized_dir, label), exist_ok=True)

for file_name in os.listdir(labelled_image_folder):
    if file_name.endswith('.jpg'):
        base_id = file_name.split('_')[0]
        if base_id in classification_df['NACCID'].astype(str).values:
            classification_type = classification_df.loc[classification_df['NACCID'].astype(str) == base_id, 'Type'].values[0]
            target_folder = os.path.join(organized_dir, classification_type)
            shutil.move(os.path.join(labelled_image_folder, file_name), os.path.join(target_folder, file_name))

# Split dataset into train, validate, and test sets
print("Splitting dataset...")
split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
for split in split_ratios:
    for label in alzheimers_category.values():
        os.makedirs(os.path.join(split_dir, split, label), exist_ok=True)

for label in alzheimers_category.values():
    label_dir = os.path.join(organized_dir, label)
    images = [f for f in os.listdir(label_dir) if f.endswith('.jpg')]
    if not images:
        continue
    train_files, temp_files = train_test_split(images, test_size=(1 - split_ratios['train']), random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=split_ratios['test'] / (split_ratios['val'] + split_ratios['test']), random_state=42)
    for split, split_files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
        for file_name in split_files:
            shutil.copy(os.path.join(label_dir, file_name), os.path.join(split_dir, split, label, file_name))

# Step 2: Define AlzhiNet model
class AlzhiNet(nn.Module):
    def __init__(self, num_classes):
        super(AlzhiNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Placeholder for dynamically determining the size
        self.flattened_size = None

        self.fc1 = None  # Fully connected layer will be initialized later
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the tensor
        x = torch.flatten(x, 1)

        # Dynamically initialize the first fully connected layer
        if self.fc1 is None:
            self.flattened_size = x.shape[1]
            self.fc1 = nn.Linear(self.flattened_size, 1024).to(x.device)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 3: Train the model
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

epochs = 10
for epoch in range(epochs):
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
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluate the model
print("Evaluating the model...")
test_dataset = datasets.ImageFolder(root=os.path.join(split_dir, 'test'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

print(f"Test Accuracy: {100 * correct / total:.2f}%")

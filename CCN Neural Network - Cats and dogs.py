import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Step 1: Define transformations
transform = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Step 2: Load the training dataset
trainset = torchvision.datasets.ImageFolder(root='PATH_TO_TRAINING_DATASET', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Step 3: Define the CNN model
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Input: 3 channels, Output: 16 channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Input: 16 channels, Output: 32 channels
        self.fc1 = nn.Linear(32 * 45 * 45, 128)  # Adjusted based on output size after pooling
        self.fc2 = nn.Linear(128, 2)  # 128 neurons, 2 classes: cat and dog

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten to match the output size
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 4: Initialize model, loss function, and optimizer
model = CatDogCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Training loop
for epoch in range(5):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Print progress
    print(f'Epoch [{epoch + 1}/25], Loss: {running_loss / len(trainloader):.4f}')

# Step 6: Visualize predictions on the training set
# Get random images and predictions from the training set
dataiter = iter(trainloader)
images, labels = next(dataiter)

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

# Plot the images and predictions
fig, axes = plt.subplots(4, 5, figsize=(12, 8))
for i in range(20):
    ax = axes[i // 5, i % 5]
    ax.imshow(images[i].permute(1, 2, 0).numpy())  # Convert from (C, H, W) to (H, W, C)
    ax.set_title(f'Predicted: {"Dog" if predicted[i] == 1 else "Cat"}')
    ax.axis('off')
plt.show()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# 1. Load and preprocess dataset
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# -----------------------------
# 2. Define CNN model
# -----------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

# -----------------------------
# 3. Loss and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# -----------------------------
# 4. Training loop
# -----------------------------
print("Training...")
for epoch in range(5):  # 5 epochs = good accuracy (~98-99%)
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")

# -----------------------------
# 5. Testing
# -----------------------------
print("\nEvaluating...")
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")

# -----------------------------
# 6. Visualize predictions
# -----------------------------
dataiter = iter(testloader)
images, labels = next(dataiter)
outputs = net(images)
_, preds = torch.max(outputs, 1)

for i in range(5):
    plt.imshow(images[i][0], cmap="gray")
    plt.title(f"Label: {labels[i]}, Predicted: {preds[i]}")
    plt.show()

# -----------------------------
# 7. Interactive user input loop
# -----------------------------
user_transform = transforms.Compose([
    transforms.Grayscale(),        # ensure 1 channel
    transforms.Resize((28, 28)),   # resize to MNIST size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("\nInteractive mode: upload your own digit images!")
print("Type 'quit' to exit.\n")

while True:
    filepath = input("Enter path to image file (or 'quit' to exit): ").strip()
    if filepath.lower() == "quit":
        print("Exiting interactive mode.")
        break

    try:
        img = Image.open(filepath)
        img_tensor = user_transform(img).unsqueeze(0)  # add batch dimension

        with torch.no_grad():
            output = net(img_tensor)
            _, pred = torch.max(output, 1)

        plt.imshow(img, cmap="gray")
        plt.title(f"Predicted: {pred.item()}")
        plt.show()

    except Exception as e:
        print(f"Error loading file: {e}")
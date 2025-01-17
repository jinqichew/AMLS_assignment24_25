import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def cnn_A():
    print("\nTask A CNN:")

    # load data
    data = np.load("Datasets/breastmnist.npz")
    train_images = data['train_images']
    train_labels = data['train_labels']
    val_images = data['val_images']
    val_labels = data['val_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    # preprocess the data
    # convert np arrays to torch
    # normalize image data by dividing 255
    # add a channel dimension (grayscale images need single channel)
    X_train = torch.from_numpy(train_images).float().unsqueeze(1) / 255.0
    y_train = torch.from_numpy(train_labels).long().squeeze()
    X_val = torch.from_numpy(val_images).float().unsqueeze(1) / 255.0
    y_val = torch.from_numpy(val_labels).long().squeeze()
    X_test = torch.from_numpy(test_images).float().unsqueeze(1) / 255.0
    y_test = torch.from_numpy(test_labels).long().squeeze()

    # combine image and label into a pytorch dataset object
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # create batch and shuffle training data
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # define model
    class CNN_taskA(nn.Module):
        def __init__(self):
            super(CNN_taskA, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc = nn.Linear(32 * 7 * 7, 2)

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = x.view(-1, 32 * 7 * 7)
            x = self.fc(x)
            return x

    model = CNN_taskA()

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 30
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # training Loop
        model.train()
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_accuracy = correct_train/total_train
        train_accuracies.append(train_accuracy)
        train_losses.append(loss.item())

        # evalutaion loop on validation set
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss / len(val_loader):.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # plot loss and accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Over Epochs')

    plt.tight_layout()
    plt.show()

    # evaluation on test set
    model.eval()
    test_preds = []
    with torch.no_grad():
        for images, _ in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.tolist())

    test_accuracy = accuracy_score(y_test.numpy(), test_preds)
    print(f"Task A CNN Test Accuracy: {test_accuracy:.4f}")

    print(classification_report(y_test.numpy(), test_preds))
    print(confusion_matrix(y_test.numpy(), test_preds))

# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

To classify the given images into it's category

## Neural Network Model

<img width="962" height="468" alt="image" src="https://github.com/user-attachments/assets/9dd756d3-a1f3-48c3-b2bc-30ac4717949a" />


## DESIGN STEPS
### STEP 1: Problem Statement
Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

### STEP 2:Dataset Collection
Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.
### STEP 3: Data Preprocessing
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.
### STEP 4:Model Architecture
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.
### STEP 5:Model Training
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.
### STEP 6:Model Evaluation
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.
### STEP 7: Model Deployment & Visualization
Save the trained model, visualize predictions, and integrate it into an application if needed.


## PROGRAM

### Name: Kannan S
### Register Number: 212223230098
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2= nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3= nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1=nn.Linear(128*3*3, 128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



```

```python
# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name: Kannan S')
        print('Register Number: 212223230098')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

<img width="326" height="202" alt="image" src="https://github.com/user-attachments/assets/192820af-d837-440c-8fd6-3d7d9354c36d" />


### Confusion Matrix

<img width="779" height="697" alt="image" src="https://github.com/user-attachments/assets/ef286387-4d79-4104-8b46-98b9829ed213" />


### Classification Report

<img width="466" height="345" alt="image" src="https://github.com/user-attachments/assets/15fb0396-c289-4ac7-a044-f58fa3fae275" />



### New Sample Data Prediction

<img width="406" height="496" alt="image" src="https://github.com/user-attachments/assets/e7811299-390e-4902-8608-999c87db5210" />


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.

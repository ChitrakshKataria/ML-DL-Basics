import torch
from torch import nn
import numpy as np
import torch_directml
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split



### Device agnostic code
device = torch_directml.device() if torch_directml.device_count() > 0 else "cpu"
#print(f"Using device: {torch_directml.device_name(0)}")

### Reproducebility
np.random.seed(42)
torch.manual_seed(42)

### Fetching the data
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
## Normalising the data
X = X/255
y = np.astype(y, np.dtype(int))

### np -> PyTorch Tensor
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).long()
print(X, y)

### Training and testing split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=784, out_features=784)
        self.activatoin1 = nn.ReLU()

        self.layer2 = nn.Linear(in_features=784, out_features=24)
        self.activatoin2 = nn.ReLU()

        self.layer3 = nn.Linear(in_features=24, out_features=24)
        self.activatoin3 = nn.ReLU()

        self.OutputLayer = nn.Linear(in_features=24, out_features=10)
        self.SoftMax = nn.Softmax(dim=1) # I don't use this yet but its here for a future project.
    def forward(self, x: torch.tensor):
        x = self.layer1(x)
        x = self.activatoin1(x)

        x = self.layer2(x)
        x = self.activatoin2(x)

        x = self.layer3(x)
        x = self.activatoin3(x)

        x = self.OutputLayer(x)
        return x

# Init Model
model1 = LinearModel().to(device)

### Training loop
epochs = 20
learningRate = 0.1 
batch_size = 50


## Dfining the loss func and opptimizer
optimizer = torch.optim.SGD(model1.parameters(), lr=learningRate)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1, epochs+1):
    epoch_loss = 0
    total_samples = 0

    ### Shuffle the data 
    indecies = torch.randperm(len(X_train))
    X_train = X_train[indecies]
    y_train = y_train[indecies]

    for start in range(0, len(X_train), batch_size):
        end = start + batch_size
        ## Current Batch
        X_batch = X_train[start:end].to(device)
        y_batch = y_train[start:end].to(device)

        ## Forward Propogate
        train_predictions = model1(X_batch)

        ## Calculate the loss
        loss = loss_fn(train_predictions, y_batch)

        ## Prep for the backpropogation
        optimizer.zero_grad()

        ## Backpropogate
        loss.backward()

        ## Update the old weights and baises with the new once
        optimizer.step()

        ## Calculating average loss
        batch_count = len(X_batch)
        epoch_loss += loss.item()*batch_count
        total_samples += batch_count


    print(f"------------Epoch: {epoch}-----------")
    print(f"Acerage Loss: {epoch_loss / total_samples}")


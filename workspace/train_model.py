import torch
from torch import nn

from workspace.models.model import myawesomemodel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load data 
train_images = torch.load("data/processed/train_images.pt")
train_targets = torch.load("data/processed/train_target.pt")

# Model 
model = myawesomemodel.to(device)

# Training
lr = 1e-3
batch_size = 256
num_epochs = 20
train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_images, train_targets), batch_size=batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()


for epoch in range(num_epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        y = y.to(torch.int64)
        # y = torch.zeros(batch_size, 10).scatter_(1, y.unsqueeze(1), 1).to(device)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} Loss {loss}")

torch.save(model, "workspace/models/model.pt")


# # Evaluation
# test_images = torch.load("data/processed/test_images.pt")
# test_targets = torch.load("data/processed/test_target.pt")
# model.eval()
# test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_images, test_targets), batch_size=64, shuffle=False)
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from datetime import datetime

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a simple dataset: y = 2x + 1 + noise
torch.manual_seed(42)
x = torch.randn(100, 1)
y = 2 * x + 1 + 0.1 * torch.randn(100, 1)

# Define a simple linear model
model = nn.Linear(1, 1).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Move data to device
x = x.to(device)
y = y.to(device)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
with mlflow.start_run(run_name=f"helloworld-torchrun-{timestamp}"):
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log metrics
        if (epoch + 1) % 20 == 0:
            mlflow.log_metric("loss", loss.item(), step=epoch)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Print learned parameters
print(f"\nLearned weight: {model.weight.data.item():.4f} (expected: ~2.0)")
print(f"Learned bias: {model.bias.data.item():.4f} (expected: ~1.0)")
print("Training complete!")

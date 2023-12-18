import torch

seeds = [12, 32, 42, 62, 153]

# Set a manual seed for reproducibility
torch.manual_seed(42)

# Now, any random operation using PyTorch will produce the same results on each run
random_tensor = torch.randn(3, 3)
print(random_tensor)


for seed in seeds:
    torch.manual_seed(seed)
    random_tensor = torch.randn(3, 3)
    print(random_tensor)

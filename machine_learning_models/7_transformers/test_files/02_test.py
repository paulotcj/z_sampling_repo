import torch

# Example tensor of shape [64, 1, 1, 100] filled with True/False values
tensor = torch.randint(0, 2, (64, 1, 1, 100), dtype=torch.bool)

# Count True values
num_true = torch.sum(tensor)

# Count False values
num_false = tensor.numel() - num_true

print(f"Number of True values: {num_true.item()}")
print(f"Number of False values: {num_false.item()}")
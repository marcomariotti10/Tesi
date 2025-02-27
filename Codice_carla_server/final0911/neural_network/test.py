import torch

import numpy as np

from scipy.optimize import linear_sum_assignment

from torch.nn import functional as F

seed = 46
np.random.seed(seed)

# Generate 32 random 2D points
pred_points = np.random.randint(0, 25, size=(20, 2))

# Shuffle to create a different order
target_points = np.random.permutation(pred_points)

target_points[0][0] += 1

print("Original points:\n", pred_points)
print("\nShuffled points:\n", target_points)

# Convert to PyTorch tensors
pred_points = torch.tensor(pred_points, dtype=torch.float32)
target_points = torch.tensor(target_points, dtype=torch.float32)

dist_matrix = torch.cdist(pred_points, target_points, p=2)  # Shape: (32, 32)
print(dist_matrix)
# Solve the linear sum assignment problem (Hungarian algorithm)
row_ind, col_ind = linear_sum_assignment(dist_matrix.cpu().detach().numpy())

# Compute the loss for the optimal assignment
optimal_pred_points = pred_points[row_ind]
optimal_target_points = target_points[col_ind]
print("\nOptimal assignment:\n", optimal_pred_points)
print("\noptimal target:\n", optimal_target_points)
for p in range(4):
    print("Distance between", optimal_pred_points[p], "and", optimal_target_points[p], ":", torch.dist(optimal_pred_points[p], optimal_target_points[p]))
loss = F.mse_loss(optimal_pred_points, optimal_target_points)

print("\nOptimal assignment loss:", loss.item())

total_loss = 0
total_loss += loss

print("\nTotal loss:", total_loss)
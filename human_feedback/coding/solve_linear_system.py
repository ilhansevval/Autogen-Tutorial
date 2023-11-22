# filename: solve_linear_system.py
import numpy as np

# Coefficients matrix A
A = np.array([
    [1, 0, 1],  # Coefficients for a, b, c in the first equation
    [1, 1, 0],  # Coefficients for a, b, c in the second equation
    [0, 1, 1]   # Coefficients for a, b, c in the third equation
])

# Constants vector B
B = np.array([1, 2, 4])  # Constants from the right side of the equations

# Solve for X (variables a, b, c)
X = np.linalg.solve(A, B)

# Calculate the sum of a, b, and c
sum_abc = np.sum(X)

# Output the result
print(f"The sum of a, b, and c is: {sum_abc}")
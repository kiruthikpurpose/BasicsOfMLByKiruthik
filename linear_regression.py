# Program to demonstrate Linear Regression

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Generating random data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Plotting the data
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Data')
plt.show()

# Implementing Linear Regression using the normal equation method
X_b = np.c_[np.ones((100, 1)), X]  # Add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Printing the parameters (theta0 and theta1)
print("Theta0:", theta_best[0][0])
print("Theta1:", theta_best[1][0])

# Predicting using the model
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)

# Plotting the predictions
plt.plot(X_new, y_predict, "r-")
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Prediction')
plt.show()

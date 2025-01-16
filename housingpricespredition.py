import numpy as np
import matplotlib.pyplot as plt






# Generate synthetic data
def generate_data(num_samples=100, novariance = 1.0):
    np.random.seed(0)
    X = 2 * np.random.rand(num_samples, 1)  #housing sizes
    y = 4 + 3 * X + np.random.randn(num_samples, 1) * novariance  # Prices
    return X, y

# Compute of  cost function
def compute_cost(X, y, theta):
    m =len(y)  # Number of samples
    predictions = X.dot(theta)  # the predictions
    cost = (1 / (2* m)) * np.sum((predictions-y) ** 2)  
    return cost






# Gradient descent implementation
def gradient_descent(X, y, learning_rate=0.1, n_iterations=1000):
    m = len(y)  # Number of samples
    theta = np.random.randn(2, 1)  
    X_b = np.c_[np.ones((m, 1)), X]  
    cost_history = []  # Store thecost values
    





    for iteration in range(n_iterations):
        gradients = 2/m*X_b.T.dot(X_b.dot(theta) - y)  # Compute gradients
        theta -= learning_rate * gradients  # Update parameters
        cost = compute_cost(X_b, y, theta)  # Compute Cost
        cost_history.append(cost)  
        
    return theta, cost_history

# Training the model
def train_model(X, y, learning_rate=0.1, n_iterations=1000):
    theta_best, cost_history = gradient_descent(X, y, learning_rate, n_iterations)
    return theta_best, cost_history


def plot_cost_history(cost_history):
    plt.plot(cost_history, "b-")  # Plot thhe cost
    plt.title("Cost Function Over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.grid()
    plt.show()
"""
def plot_results(X, y, theta):
    plt.plot(X, y, "b.", label="Training data")  # Training data
    X_new = np.array([[0], [2]])  # New data for predictions
    X_new_b= np.c_[np.ones((2, 1)), X_new]  # add intercept
    y_predict = X_new_b.dot(theta)  # Predictions
    plt.plot(X_new, y_predict, "r-", label="Predictions")  # Predictions
    plt.xlabel("Size of the house")
    plt.ylabel("Price of the house")
    plt.title("Housing Prices Prediction using Gradient Descent")
    plt.legend()
    plt.show()

# Main execution flow
if __name__ == "__main__":
    X, y = generate_data(num_samples=100, noise_variance=1.0)  # Generate data
    theta_best, cost_history = train_model(X, y, learning_rate=0.1, n_iterations=1000)  # Train model
    print("Estimated parameters (intercept and slope):", theta_best.ravel())  # Print parameters
    plot_cost_history(cost_history)  # Plot cost history
    plot_results(X, y, theta_best)  # Plot results
# ize results



"""def plot_results(X, y, theta):
    plt.plot(X, y, "b.", label="Training data")  # Training data
    X_new= np.array([[0], [2]])  # New data for predictions
    X_new_b= np.c_[np.ones((2, 1)), X_new]  # add intercept
    y_predict= X_new_b.dot(theta)  # Predictions
    plt.plot(X_new, y_predict, "r-", label="Predictions")  #  predictions
    plt.xlabel("Size of the house")
    plt.ylabel("Price of the house")
    plt.title("Housing Prices Prediction using Gradient Descent")
    plt.legend()
    plt.show()

# Main  flow
if __name__ == "__main__":
    X, y = generate_data(num_samples=100, noise_variance=1.0)  # Generate data
    theta_best, cost_history = train_model(X, y, learning_rate=0.1, n_iterations=1000)  # Train model
    print("Estimated parameters (intercept and slope):", theta_best.ravel())  # Print parameters
    plot_cost_history(cost_history)  # Plot cost history
    plot_results(X, y, theta_best)  # Plot results

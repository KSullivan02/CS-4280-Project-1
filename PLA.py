import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Generate Linear Data
def generate_linearly(n_points=15):
    # class 1
    class1 = np.random.randn(n_points, 2) + np.array([2,2])

    # class -1
    class2 = np.random.randn(n_points, 2) + np.array([-2,-2])
    X = np.vstack((class1,class2))
    y = np.hstack((np.ones(n_points), -np.ones(n_points)))

    return X, y

# Generate Non-Linear Data
def generate_non_linearly(n_points=15):
    # class 1
    class1 = np.random.randn(n_points, 2) + np.array([1,1])

    # class 2
    class2 = np.random.randn(n_points, 2) + np.array([1,-1])

    X = np.vstack((class1, class2))
    y = np.hstack((np.ones(n_points), -np.ones(n_points)))
    return X, y

def perceptron(X, y, eta=1.0, epochs=200):
    X = np.c_[np.ones(X.shape[0]), X] # grants bias
    weights = np.zeros(X.shape[1]) #Initial weights of 0
    updates = 0

    # for the number of epochs (iterations) go through the list
    for _ in range(epochs):
        error = False
        for i in range(X.shape[0]):
            if np.sign(np.dot(weights, X[i])) != y[i]:
                weights += eta * y[i] * X[i]
                updates += 1
                error = True
        if not error:
            break

    # Calculate training error
    predictions = np.sign(np.dot(X, weights))
    training_error = np.mean(predictions != y) * 100

    return weights, updates, training_error

# plot the boundary line
def plot_bound(weights, X, y, title):
    plt.figure(figsize=(8,6))
    plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolors='k')
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    x_vals = np.linspace(xmin, xmax, 100)
    y_vals = -(weights[0] + weights[1] * x_vals) / weights[2]
    plt.plot(x_vals, y_vals, 'k--')

    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True)
    plt.show()

# generate linear data for training
train_X_lin, train_y_lin = generate_linearly()
df_lin = pd.DataFrame(train_X_lin)
df_lin.to_csv("linearly.csv")

# generate non-linear training data
train_X_non, train_y_non = generate_non_linearly()
df_non = pd.DataFrame(train_X_non)
df_non.to_csv("non_linearly.csv")

# generate test data
test_X, test_y = generate_linearly(5) # Test data should remain lineraly sperable to show loss of accuracy from less sorted training data

# Testing Linear
weights_linear, updates_linear, training_err_lin = perceptron(train_X_lin, train_y_lin)
plot_bound(weights_linear, train_X_lin, train_y_lin, "Lineraly Seperable Data Training")

# Display training results for linearly separable data
print(f"Linearly Separable Training Data:")
print(f"  Number of weight updates: {updates_linear}")
print(f"  Training error rate: {training_err_lin:.2f}%")

# Testing Non-Linear
weights_non, updates_non, training_err_non = perceptron(train_X_non, train_y_non)
plot_bound(weights_non, train_X_non, train_y_non, "Non-Lineraly Speperable Data Training")

print(f"Non-Linearly Separable Training Data:")
print(f"  Number of weight updates: {updates_non}")
print(f"  Training error rate: {training_err_non:.2f}%")

test_X_aug = np.c_[np.ones(test_X.shape[0]), test_X]
predictions = np.sign(np.dot(test_X_aug, weights_linear))

# Calc Accuracy and Classification Support
accuracy = accuracy_score(test_y, predictions)
print(accuracy)

report = classification_report(predictions, test_y, digits=2)
print(report)


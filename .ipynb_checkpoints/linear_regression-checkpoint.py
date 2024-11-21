# Iris dataset (first feature: sepal length, second feature: sepal width)
iris_data = [
    [5.1, 3.5], [4.9, 3.0], [4.7, 3.2], [4.6, 3.1], [5.0, 3.6], [5.4, 3.9], [4.6, 3.4], [5.0, 3.4], [4.4, 2.9], 
    [4.9, 3.1], [5.4, 3.7], [4.8, 3.4], [4.8, 3.0], [4.3, 3.0], [5.8, 4.0], [5.7, 4.4], [5.4, 3.9], [5.1, 3.5],
    [5.7, 3.8], [5.1, 3.8], [5.4, 3.4], [5.1, 3.7], [4.6, 3.6], [5.1, 3.3], [4.8, 3.4], [5.0, 3.0], [5.0, 3.4],
    [5.2, 3.5], [5.2, 3.4], [4.7, 3.2]
]

# Split the data into input (X) and output (y)
X = [row[0] for row in iris_data]
y = [row[1] for row in iris_data]

# Normalize the data
def normalize(data):
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = variance ** 0.5
    return [(x - mean) / std_dev for x in data]

X = normalize(X)
y = normalize(y)

# Split into training and testing sets (80-20 split)
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Linear Regression using Gradient Descent
def linear_regression(X, y, learning_rate=0.01, iterations=1000):
    m = 0  # Slope
    c = 0  # Intercept
    n = len(X)
    
    for _ in range(iterations):
        # Predictions
        y_pred = [m * X[i] + c for i in range(n)]
        # Gradients
        dm = (-2 / n) * sum((y[i] - y_pred[i]) * X[i] for i in range(n))
        dc = (-2 / n) * sum((y[i] - y_pred[i]) for i in range(n))
        # Update parameters
        m -= learning_rate * dm
        c -= learning_rate * dc
    
    return m, c

# Train the model
m, c = linear_regression(X_train, y_train)

# Predictions
def predict(X, m, c):
    return [m * x + c for x in X]

y_pred_train = predict(X_train, m, c)
y_pred_test = predict(X_test, m, c)

# Calculate Mean Squared Error (MSE) and R² Score
def mse(y_true, y_pred):
    n = len(y_true)
    return sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n

def r2_score(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_total = sum((y - mean_y) ** 2 for y in y_true)
    ss_residual = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
    return 1 - (ss_residual / ss_total)

# Evaluate the model
train_mse = mse(y_train, y_pred_train)
test_mse = mse(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")
print(f"Train R² Score: {train_r2}, Test R² Score: {test_r2}")
print(f"Model Coefficients: m = {m}, c = {c}")

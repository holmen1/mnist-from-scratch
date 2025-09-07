import numpy as np



class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
    # Input to hidden
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)  # Activation function
        
        # Hidden to output
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)  # Output probabilities
        
        return self.a2
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss
    
    def one_hot_encode(self, labels, num_classes=10):
        one_hot = np.zeros((labels.shape[0], num_classes))
        one_hot[np.arange(labels.shape[0]), labels] = 1
        return one_hot
    
    def backward(self, X, y_true, learning_rate=0.01):
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = self.a2 - y_true
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # ReLU derivative
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X_train, y_train, epochs=100, learning_rate=0.01):
        y_train_onehot = self.one_hot_encode(y_train)
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X_train)
            
            # Compute loss
            loss = self.cross_entropy_loss(predictions, y_train_onehot)
            
            # Backward pass
            self.backward(X_train, y_train_onehot, learning_rate)
            
            # Print progress
            if epoch % 10 == 0:
                accuracy = self.accuracy(predictions, y_train)
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    def predict(self, X):
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)

    def accuracy(self, predictions, labels):
        pred_classes = np.argmax(predictions, axis=1)
        return np.mean(pred_classes == labels)
import numpy as np

class PerceptronSimple:
    def __init__(self, num_inputs, learning_rate=0.01, epochs=100):
        self.weights = np.random.rand(num_inputs + 1)  # +1 para el sesgo
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # Agregar el sesgo
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels):
        for epoch in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error

# Ejemplo de uso
training_inputs = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

labels = np.array([0, 0, 0, 0, 0, 0, 0, 1])

perceptron = PerceptronSimple(num_inputs=3)

print("Antes del entrenamiento:")
for inputs in training_inputs:
    print(f"Entrada: {inputs}, Predicción: {perceptron.predict(inputs)}")

perceptron.train(training_inputs, labels)

print("\nDespués del entrenamiento:")
for inputs in training_inputs:
    print(f"Entrada: {inputs}, Predicción: {perceptron.predict(inputs)}")

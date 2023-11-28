class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1, epochs=100):
        # Inicializamos pesos y umbral de manera aleatoria
        self.weights = [0.0] * num_inputs
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, inputs):
        # Calculamos la suma ponderada
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        # Aplicamos la función de activación escalón unitario
        return 1 if weighted_sum >= 0 else 0

    def train(self, training_data):
        for epoch in range(self.epochs):
            for inputs, target in training_data:
                # Realizamos la predicción
                prediction = self.predict(inputs)
                # Calculamos el error
                error = target - prediction
                # Actualizamos pesos y umbral según la regla de aprendizaje del perceptrón
                self.weights = [w + self.learning_rate * error * x for w, x in zip(self.weights, inputs)]
                self.bias += self.learning_rate * error

# Ejemplo de entrenamiento del perceptrón para la operación lógica AND
training_data_and = [((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)]

# Creamos un perceptrón con 2 entradas
perceptron_and = Perceptron(num_inputs=2)

# Entrenamos el perceptrón
perceptron_and.train(training_data_and)

# Probamos el perceptrón entrenado
print(perceptron_and.predict((0, 0)))  # Salida esperada: 0
print(perceptron_and.predict((0, 1)))  # Salida esperada: 0
print(perceptron_and.predict((1, 0)))  # Salida esperada: 0
print(perceptron_and.predict((1, 1)))  # Salida esperada: 1


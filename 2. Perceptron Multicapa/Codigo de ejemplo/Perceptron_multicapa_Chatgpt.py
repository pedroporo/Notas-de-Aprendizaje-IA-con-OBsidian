import numpy as np

# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Datos de entrada y salida
input_data = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])

output_labels = np.array([[0], [1], [1], [0]])

# Definición de la arquitectura de la red
input_size = 2
hidden_size = 4
output_size = 1

# Inicialización de pesos y sesgos
np.random.seed(42)
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

bias_hidden = np.zeros((1, hidden_size))
bias_output = np.zeros((1, output_size))

# Hiperparámetros
learning_rate = 0.1
epochs = 10000

# Entrenamiento
for epoch in range(epochs):
    # Capa de entrada a capa oculta
    hidden_input = np.dot(input_data, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    # Capa oculta a capa de salida
    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_input)

    # Calcular la pérdida
    loss = 0.5 * np.mean((predicted_output - output_labels) ** 2)

    # Retropropagación
    output_error = output_labels - predicted_output
    output_delta = output_error * sigmoid_derivative(predicted_output)

    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # Actualización de pesos y sesgos
    weights_hidden_output += hidden_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += input_data.T.dot(hidden_delta) * learning_rate

    bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    # Imprimir la pérdida cada 1000 iteraciones
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Evaluación
final_hidden_input = np.dot(input_data, weights_input_hidden) + bias_hidden
final_hidden_output = sigmoid(final_hidden_input)
final_output = np.dot(final_hidden_output, weights_hidden_output) + bias_output
final_predicted_output = sigmoid(final_output)

print("\nPredicciones finales:")
print(final_predicted_output)

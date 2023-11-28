class test:
    def __init__ (self, num_imputs,velocidadDeAprendizaje=0.1,interacciones=100):
        self.peso = [0.0] * num_imputs
        self.bias= 0.0
        self.velocidadDeAprendizaje = velocidadDeAprendizaje
        self.interacciones = interacciones
    def prediccion(self, inputs):
        peso_suma = sum(w*x for w,x in zip (self.peso,inputs))+self.bias
        return 1 if peso_suma >=0 else 0
    def entrenamiento (self, datosDelEntrenamiento):
        for interaccion in range(self.interacciones):
            for inputs, traget in datosDelEntrenamiento:
                prediccion = self.prediccion(inputs)
                error = traget - prediccion
                self.peso=[w+self.velocidadDeAprendizaje*error*x for w,x in zip (self.peso,inputs)]
                self.bias += self.velocidadDeAprendizaje*error

datosDelEntrnamiento = [((0,0),0),((0,1),0),((1,0),0),((1,1),1)]

perceptron=test(num_imputs=2)
perceptron.entrenamiento(datosDelEntrnamiento)
print("Entrenamiento AND")
for inputs in datosDelEntrnamiento:
    print(f"Entrada: {inputs}, Predicción: {perceptron.prediccion(inputs)}")
print(perceptron.prediccion((0,0)))
print(perceptron.prediccion((0,1)))
print(perceptron.prediccion((1,0)))
print(perceptron.prediccion((1,1)))
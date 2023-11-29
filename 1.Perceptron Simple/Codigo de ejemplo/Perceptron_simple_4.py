class perceptronSimpleCode:
    def __init__(self,num_inputs,velocidadDeAprendizaje=0.1,interacciones=100):
        self.peso =[0.0]*num_inputs
        self.bias=0.0
        self.velocidadDeAprendizaje=velocidadDeAprendizaje
        self.interacciones=interacciones
    def prediccion(self,inputs):
        peso_suma = sum(w*x for w,x in zip(self.peso,inputs))+self.bias
        return 1 if peso_suma >=0 else 0
    def entrenamiento(self,training_data):
        for interaccion in range(self.interacciones):
            for input in range(self.interacciones):
                prediccion= self.prediccion(inputs)
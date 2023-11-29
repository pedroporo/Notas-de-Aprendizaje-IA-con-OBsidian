class perceptronSimpleCode:
    def __init__(self,num_inputs,velocidadDeAprendizaje=0.1,interacciones=100):
        self.peso =[0.0]*num_inputs
        self.bias=0.0
        self.velocidadDeAprendizaje=velocidadDeAprendizaje
        self.interacciones=interacciones
    def prediccion(self,inputs):
        peso_suma = sum()+self.bias
class test:
    def __init__ (self, num_imputs,velocidadDeAprendizaje=0.1,interacciones=100):
        self.peso = [0.0] * num_imputs
        self.bias= 0.0
        self.velocidadDeAprendizaje = velocidadDeAprendizaje
        self.interacciones = interacciones
    def prediccion(self, inputs):
        peso_suma = sum(x*w for w,x in zip (self.peso,inputs))+self.bias
        return 1 if peso_suma >=0 else 0
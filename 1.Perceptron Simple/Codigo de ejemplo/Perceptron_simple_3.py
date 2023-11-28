class test:
    def __init__ (self, num_imputs,velocidadDeAprendizaje=0.1,interacciones=100):
        self.peso = [0.0] * num_imputs
        self.bias= 0.0
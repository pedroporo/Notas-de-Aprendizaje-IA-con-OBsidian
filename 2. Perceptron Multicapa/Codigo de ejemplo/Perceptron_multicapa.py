import numpy as np

def activacionSigmoide(x):
    return 1/(1+np.exp(-x))
def sigmoideDerivativo(x):
    return x *(1-x)

datosDeEntrada = np.array([[0,0],[0,1],[1,0],[1,1]])
capasDeSalida = np.array(([0],[1],[1],[0]))
#La cantidad de caracteres por entrada es 2 ya que el tamaño de los datos que introducimos son de 2 caracteres si ponemos mas aumentaria esta variable
cantidadDeCaracteresPorEntrada = 2
#De momento no se que hace esta variable
tamañoOculto=4
#El tamaño de slida es 1 ya que solo se va a necesitar 1 dato
tamañoDeSalida=1
#Aqui siempre generara los mismos valores aleatorios que esten en la semilla 42    
np.random.seed(42)
#Aqui se crean los pesos para la entrada a la capa oculto como minimo es el valor de la cantidad de caracteres y como maximo es el tamaño oculto
pesosDeEntradaAOculto=np.random.uniform(size=(cantidadDeCaracteresPorEntrada,tamañoOculto))
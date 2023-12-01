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
#Aqui se generan los pesos de la capa oculta a la capa de salida
pesosDeOcultoASalida=np.random.uniform(size=(tamañoOculto,tamañoDeSalida))
#Esto de aqui hace un array con valor 0 y su forma es asi:
#[         ][Columnas=1]
#[Filas = 4][          ]
#[Filas = 4][          ]
#[Filas = 4][          ]
#[Filas = 4][          ]
classificadorOculto=np.zeros(1,tamañoOculto)
#Esto de aqui hace un array con valor 0 y su forma es asi:
#[         ][Columnas=1]
#[Filas = 1][          ]
classificadorSalida=np.zeros(1,tamañoDeSalida)
#La velocidad de aprendizaje es lo rapido que ira aprendiendo la ia, lo recomendable es ajustarlo a las necesidades que tenemos
# (si es pequeño es mas preciso y si es mas grande es menos preciso)
velocidadDeAprendizaje=0.1
#Esto son el numero de vezes que se va a entrenar la inteligencia artificial
interacciones=10000
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
classificadorOculto=np.zeros((1,tamañoOculto))
#Esto de aqui hace un array con valor 0 y su forma es asi:
#[         ][Columnas=1]
#[Filas = 1][          ]
classificadorSalida=np.zeros((1,tamañoDeSalida))
#La velocidad de aprendizaje es lo rapido que ira aprendiendo la ia, lo recomendable es ajustarlo a las necesidades que tenemos
# (si es pequeño es mas preciso y si es mas grande es menos preciso)
velocidadDeAprendizaje=0.1
#Esto son el numero de vezes que se va a entrenar la inteligencia artificial
interacciones=10000

#Aqui se hace el entrenamiento de la inteligencia artificial
for interaccion in range(interacciones):

    entrada_oculta = np.dot(datosDeEntrada,pesosDeEntradaAOculto)+classificadorOculto
    salida_oculta = activacionSigmoide(entrada_oculta)
    
    datos_de_entrada_para_salida=np.dot(salida_oculta,pesosDeOcultoASalida)+classificadorSalida
    salida_predicha = activacionSigmoide(datos_de_entrada_para_salida)
    #Aqui se calcula la perdida que tiene las respuestas haciendo 0.5 por la salida predicha - las capas de salida elevadas por 2
    perdida = 0.5 * np.mean((salida_predicha-capasDeSalida)** 2)

    errorDeSalida = capasDeSalida-salida_predicha
    deltaSalida = errorDeSalida * sigmoideDerivativo(salida_predicha)

    errorOculto = deltaSalida.dot(pesosDeOcultoASalida.T)
    deltaOculta = errorOculto * sigmoideDerivativo(salida_oculta)

    pesosDeOcultoASalida += salida_oculta.T.dot(deltaSalida)*velocidadDeAprendizaje
    pesosDeEntradaAOculto += datosDeEntrada.T.dot(deltaOculta)*velocidadDeAprendizaje

    classificadorSalida+=np.sum(deltaSalida,axis=0,keepdims=True)*velocidadDeAprendizaje

    classificadorOculto+=np.sum(deltaOculta,axis=0,keepdims=True)*velocidadDeAprendizaje

    if interaccion % 1000 ==0:
        print(f'Interaccion {interaccion}, Perdida: {perdida}')

entrada_oculta_final=np.dot(datosDeEntrada,pesosDeEntradaAOculto)+classificadorOculto
salida_oculta_final=activacionSigmoide(entrada_oculta_final)
salida_final=np.dot(salida_oculta_final,pesosDeOcultoASalida)+classificadorSalida
salida_predicha_final=activacionSigmoide(salida_final)

print("\nPredicciones finales")
print(salida_predicha_final)
testeo='Hola como estas'.split()
print(testeo)
frase = ['hola','como','estas','pedro']
print(frase)
for frase in range(frase.count):
    print(frase)
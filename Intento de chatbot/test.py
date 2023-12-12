import numpy as np
import json

intents=json.loads(open('/home/batoi/Obsidian Vaults/Notas_De_Aprendizaje_IA_con_Obsidian/Notas-de-Aprendizaje-IA-con-OBsidian/Intento de chatbot/intents.json').read())

for intent in intents['intents']:
    for pattern in intent['patterns']:
        print(pattern.split())
print('Respuestas')
for intent in intents['intents']:
    for response in intent['responses']:
        print(response.split())


def activacionSigmoide(x):
    return 1/(1+np.exp(-x))
def sigmoideDerivativo(x):
    return x *(1-x)

datosDeEntrada = np.array([[0,0],[0,1],[1,0],[1,1]])
capasDeSalida = np.array(([0],[1],[1],[0]))

cantidadDeCaracteresPorEntrada = 2
tamañoOculto=4
tamañoDeSalida=1
np.random.seed(42)
pesosDeEntradaAOculto=np.random.uniform(size=(cantidadDeCaracteresPorEntrada,tamañoOculto))
pesosDeOcultoASalida=np.random.uniform(size=(tamañoOculto,tamañoDeSalida))
classificadorOculto=np.zeros((1,tamañoOculto))
classificadorSalida=np.zeros((1,tamañoDeSalida))
velocidadDeAprendizaje=0.1
interacciones=10000


for interaccion in range(interacciones):

    entrada_oculta = np.dot(datosDeEntrada,pesosDeEntradaAOculto)+classificadorOculto
    salida_oculta = activacionSigmoide(entrada_oculta)
    
    datos_de_entrada_para_salida=np.dot(salida_oculta,pesosDeOcultoASalida)+classificadorSalida
    salida_predicha = activacionSigmoide(datos_de_entrada_para_salida)
    
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
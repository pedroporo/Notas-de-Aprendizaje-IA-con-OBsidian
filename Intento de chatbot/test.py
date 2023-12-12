import numpy as np
import json

intents=json.loads(open('/home/batoi/Obsidian Vaults/Notas_De_Aprendizaje_IA_con_Obsidian/Notas-de-Aprendizaje-IA-con-OBsidian/Intento de chatbot/intents.json').read())

for intent in intents['intents']:
    for pattern in intent['patterns']:
        print(pattern.split())

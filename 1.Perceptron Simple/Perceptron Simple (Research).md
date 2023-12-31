## Componentes principales del Perceptrón

El perceptrón está formado por una serie de componentes como:

- **Entrada.** Las entradas en el algoritmo del perceptrón se entienden como $x_1, x_2, x_3, x_4$ y así sucesivamente. Todas estas entradas denotan los valores del perceptrón de características y la ocurrencia total de las características.
- **Pesos.** Se observan como valores que se planifican al largo de la sesión de preparación del perceptrón. Los pesos ofrecen un valor preliminar en el inicio del aprendizaje del algoritmo. Con la ocurrencia de cada inexactitud de entrenamiento, los valores de los pesos se actualizan. Estos se representan principalmente como $w_1, w_2, w_3, w_4$ y así sucesivamente.
- **Suma ponderada.** Es la proliferación de cada valor de entrada o característica asociada con el valor de paso correspondiente.
- **Función de activación.** Cada función de activación, o no lineal, toma un único número y realiza una determinada operación matemática fija sobre él. Hay varias funciones de activación que se pueden encontrar en la práctica, las más comunes son la Sigmoide o la ReLU o unidad lineal rectificada.
- **Salida.** La suma ponderada se pasa a la función de activación y cualquier valor que obtengamos después del cálculo es nuestra salida predicha.
![[imagenes/perceptron.jpg]]
## Modelo de perceptrón de una capa

Un modelo de perceptrón de una capa incluye una red feedforward que depende de una función de transferencia de umbral en su modelo. Es el tipo más sencillo de red neuronal artificial que puede analizar solo objetos linealmente separables con resultados binarios, es decir, 1 y 0.

Si hablamos del funcionamiento del modelo de perceptrón de una capa, su algoritmo no tiene información previa, por lo que inicialmente, los pesos se asignan de forma inconstante, entonces el algoritmo suma todas las entradas ponderadas, si el valor añadido es más que algún valor predeterminado o valor umbral, entonces el perceptrón de una capa se declara activo y entrega la salida como +1.

En palabras sencillas, los valores de entrada múltiples alimentan el modelo del perceptrón, el modelo se ejecuta con los valores de entrada, y si el valor estimado es el mismo que la salida requerida, entonces el rendimiento del modelo se encuentra satisfecho, por lo que los pesos no exigen cambios. De hecho, si el modelo no cumple con el resultado requerido, entonces se realizan algunos cambios en los pesos para minimizar los errores.
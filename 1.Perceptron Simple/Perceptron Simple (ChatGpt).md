
Un perceptrón simple es un modelo de red neuronal artificial que se utiliza para realizar clasificación binaria. Fue propuesto por Frank Rosenblatt en 1957 y es la unidad básica de procesamiento en una red neuronal de una sola capa. El perceptrón simple toma varias entradas binarias, aplica pesos a estas entradas, suma los resultados y luego aplica una función de activación para producir una salida binaria.

Matemáticamente, el proceso en un perceptrón simple se puede describir de la siguiente manera:

1. Se tienen entradas $x_1, x_2, ···, x_n$ que pueden tomar valores binarios (0 o 1).
2. Cada entrada se multiplica por un peso correspondiente $w_1, w_2, \ldots, w_n$
3. Se suman los productos de las entradas y los pesos: $z = w_1 · x_1 + w_2 · x_2 + ···· + w_n · x_n.$
4. Se aplica una función de activación. La función de activación típicamente utilizada en un perceptrón simple es la función escalón unitario, que devuelve 1 si la entrada es mayor o igual a cero, y 0 en caso contrario. La fórmula sería $$
f(z) = \begin{cases} 1, & \text{si } z \geq 0 \\ 0, & \text{si } z < 0 \end{cases}
$$

El perceptrón simple puede aprender a ajustar sus pesos durante el entrenamiento para realizar la clasificación deseada. Sin embargo, es importante tener en cuenta que un perceptrón simple solo puede aprender a clasificar conjuntos de datos linealmente separables, es decir, conjuntos de datos que pueden ser divididos por una línea recta en el espacio de entrada.

Para abordar problemas más complejos y no lineales, se utilizan redes neuronales más avanzadas, como las redes neuronales multicapa, que constan de múltiples capas de perceptrones interconectados. Estas redes pueden aprender a representar y clasificar patrones más complejos en los datos. 


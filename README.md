¿Qué es DQN?
DQN (Deep Q-Network) es un algoritmo de aprendizaje por refuerzo (reinforcement learning) que combina Q-Learning con redes neuronales profundas para aprender políticas óptimas en entornos con estados y acciones discretas. Su objetivo es entrenar un agente para tomar decisiones que maximicen una recompensa acumulada a largo plazo.

Cómo funciona:

Usa una red neuronal para aproximar la función de valor-acción Q(s,a)
, que estima la recompensa esperada al tomar la acción a en el estado s  y seguir una política óptima.
Componentes clave:

Replay Buffer: Almacena experiencias ![imagen](https://github.com/user-attachments/assets/223bc56f-566f-409e-866d-a5a7c3d1ce18)
 para muestrear aleatoriamente y romper correlaciones temporales.
Target Network: Una copia de la red principal que estabiliza los objetivos de Q-values durante el entrenamiento.
Exploración: Usa una estrategia como ![imagen](https://github.com/user-attachments/assets/73e485c0-56e6-4bbd-b7f1-3fb500c4e228)
 para equilibrar exploración (acciones aleatorias) y explotación (acciones óptimas).


Actualiza los pesos de la red minimizando la pérdida entre los Q-values predichos y los objetivos calculados (usando la ecuación de Bellman).


Por qué se usa:

Es efectivo en entornos con espacios de estados grandes (como imágenes o vectores continuos) donde los métodos tabulares de Q-Learning son inviables.
Ha demostrado éxito en tareas como juegos de Atari, robótica, y problemas de control como CartPole-v1.
Permite aprender políticas complejas sin necesidad de un modelo explícito del entorno.



¿Por qué se utiliza Gymnasium en este programa?
Gymnasium es una biblioteca de Python (sucesora de OpenAI Gym) que proporciona entornos estandarizados para desarrollar y probar algoritmos de aprendizaje por refuerzo.

Uso en el programa:

El programa usa el entorno CartPole-v1 de Gymnasium (env = gym.make("CartPole-v1")) para simular un poste equilibrado sobre un carro móvil.
CartPole-v1:

Estado: Un vector de 4 dimensiones ![imagen](https://github.com/user-attachments/assets/821fdfe6-0819-4074-9a8c-b1616aa48109)
 (posición del carro, velocidad, ángulo del poste, velocidad angular).
Acciones: 2 discretas (mover el carro a la izquierda o derecha).
Recompensa: +1 por paso mientras el poste esté equilibrado (hasta 500 pasos o hasta que falle).
Objetivo: Maximizar la recompensa acumulada manteniendo el poste vertical el mayor tiempo posible.


Gymnasium proporciona una interfaz simple para:

Resetear el entorno (env.reset()): Inicia un episodio con un estado inicial.
Ejecutar acciones (env.step(action)): Devuelve el siguiente estado, recompensa, y si el episodio terminó.
Definir espacios (env.observation_space, env.action_space): Especifica las dimensiones de estados y acciones (4 y 2 en CartPole).




Por qué Gymnasium:

Estandarización: Ofrece un entorno consistente para probar el algoritmo DQN, facilitando la comparación con otros métodos.
Simplicidad: Maneja la dinámica del entorno (física de CartPole) sin que el programador deba implementarla.
Flexibilidad: Permite integrar el entorno con el código C++ (a través de Pybind11) y Python, como en tu programa, donde el agente DQN interactúa con CartPole para aprender a equilibrar el poste.
Depuración: Proporciona métricas claras (recompensas, estados) para evaluar el rendimiento del agente (e.g., promedio de 534.4 en el episodio 1500).



Contexto en el programa

DQN: Implementado en hdqnt.cpp (DQNModel, DQNService) para aprender una política que equilibre el poste. Usa una red neuronal, replay buffer, y target network para optimizar los Q-values.
Gymnasium: Usado en dqn_test.py para:

Crear el entorno CartPole-v1.
Generar trayectorias (estados, acciones, recompensas) que alimentan el entrenamiento del DQN.
Evaluar el rendimiento del agente mediante recompensas acumuladas (e.g., 573.53 en el episodio 1000).


Integración: El código C++ (DQN) se conecta con Python (Gymnasium) vía Pybind11, permitiendo un entrenamiento eficiente en C++ mientras se usa la interfaz de Gymnasium para el entorno.

Resumen final

DQN: Es el cerebro del agente, aprendiendo a predecir qué acciones maximizan la recompensa en CartPole mediante una red neuronal y técnicas como replay buffer y target network.
Gymnasium: Es el "mundo" donde el agente aprende, proporcionando el entorno CartPole-v1 para simular la tarea de equilibrar el poste y generar datos de entrenamiento.
Resultado: El programa logra un promedio de ~534.4, indicando que el DQN, entrenado con Gymnasium, equilibra el poste durante los 500 pasos máximos de manera consistente.

import dqn_module
import gymnasium as gym
import numpy as np

def main():
    # Configurar entorno y servicio
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    learning_rate = 0.0002
    dqn = dqn_module.DQNService(state_size, action_size, learning_rate)

    # Hiperparámetros
    episodes = 1500
    max_steps = 500
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.999
    reward_window = []

    # Límites ajustados para normalización
    state_bounds = np.array([2.4, 5.0, 0.21, 5.0])  # [x, v, theta, omega]

    # Entrenamiento
    for episode in range(episodes):
        state, _ = env.reset()
        state = np.clip(state / state_bounds, -1.0, 1.0).tolist()
        state = dqn_module.State(state)
        states = []
        actions = []
        rewards = []
        dones = []
        total_reward = 0

        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-episode / epsilon_decay)

        for _ in range(max_steps):
            action = dqn.predict(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action.id)
            next_state = np.clip(next_state / state_bounds, -1.0, 1.0).tolist()
            # Recompensa modificada más suave
            reward += 0.15 * (1.0 - abs(next_state[2] / 0.21))
            next_state = dqn_module.State(next_state)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done or truncated)
            total_reward += reward

            state = next_state
            if done or truncated:
                states.append(state)
                break

        print(f"Episodio {episode + 1}: Tamaños: states={len(states)}, actions={len(actions)}, rewards={len(rewards)}, dones={len(dones)}")
        dqn.train(states, actions, rewards, dones, gamma, learning_rate)
        
        reward_window.append(total_reward)
        if len(reward_window) > 100:
            reward_window.pop(0)
        avg_reward = np.mean(reward_window)
        
        print(f"Episodio {episode + 1}/{episodes}, Recompensa: {total_reward:.2f}, Promedio (100 ep.): {avg_reward:.1f}, Epsilon: {epsilon:.3f}")

    # Predicción
    state, _ = env.reset()
    state = np.clip(state / state_bounds, -1.0, 1.0).tolist()
    state = dqn_module.State(state)
    action = dqn.predict(state, 0.0)
    print(f"Acción predicha para estado {state.features}: {action.id}")

    env.close()

if __name__ == "__main__":
    main()
import numpy as np
import random
from typing import Tuple, List
import time

BOARDSIZE = 10
OBSTACLES = [3,24,56,82]

class GridWorld:
    def __init__(self, size: int = 5, obstacles = []):
        self.size = size
        self.state = 0  # Start position (top-left)
        self.goal = size * size - 1  # Bottom-right corner
        # Actions: 0: right, 1: down, 2: left, 3: up
        self.action_space = [0, 1, 2, 3]
        self.reset()
        self.obstacles = obstacles
        
    def reset(self) -> int:
        self.state = 0
        return self.state
        
    def step(self, action: int) -> Tuple[int, float, bool]:
        row = self.state // self.size
        col = self.state % self.size
        
        # Calculate new position based on action
        if action == 0:  # Right
            col = min(col + 1, self.size - 1)
        elif action == 1:  # Down
            row = min(row + 1, self.size - 1)
        elif action == 2:  # Left
            col = max(col - 1, 0)
        elif action == 3:  # Up
            row = max(row - 1, 0)

        new_state = row * self.size + col
            
        # Update state
        if new_state not in self.obstacles:
            self.state = new_state
        
        # Calculate reward and check if done
        done = (self.state == self.goal)
        reward = 50.0 if done else -0.1
        
        return self.state, reward, done

    def render(self):
        """Visualize the current state of the grid world."""
        print("\n" + "═" * (self.size * 4 + 1))
        for row in range(self.size):
            print("║", end=" ")
            for col in range(self.size):
                pos = row * self.size + col
                if pos == self.state:
                    print("A", end=" ║ ")  # A for Agent
                elif pos == self.goal:
                    print("G", end=" ║ ")  # G for Goal
                elif pos in self.obstacles:
                    print("O", end=" ║ ")  # O for obstacle
                else:
                    print("·", end=" ║ ")  # · for empty space
            print("\n" + "═" * (self.size * 4 + 1))

class QLearningAgent:
    def __init__(self, state_size: int, action_size: int):
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = 0.5
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.01
        
    def get_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q_table[state])
        
    def learn(self, state: int, action: int, reward: float, next_state: int, done: bool, show=False):
        if not done:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        else:
            target = reward
        if show:
            print("Target: ", target)
            print("Current Q table:")
            show_q_table(self.q_table, state)
            print("Future Q table:")
            show_q_table(self.q_table, next_state)
            
            
            
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])
        if show:
            print("Updated Q table:")
            show_q_table(self.q_table, state)
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def show_q_table(q_table, state):
    print(f"  Up: {q_table[state][3]}")
    print(f"Left: {q_table[state][2]}       Right: {q_table[state][0]}")
    print(f"  Down: {q_table[state][1]}")

def train_agent(agent, episodes: int = 1000, render: bool = False) -> List[float]:
    env = GridWorld(BOARDSIZE, OBSTACLES)
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if render:
                env.render()
                input()  # Add delay to make visualization readable
                print("\033[H\033[J", end="")  # Clear screen (works in most terminals)
                
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done, show=render)
            state = next_state
            total_reward += reward
            
        if render:
            env.render()  # Show final state
            print(f"Episode {episode + 1} finished with reward: {total_reward:.2f}")
            time.sleep(1)
            
        rewards_history.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
            
    return rewards_history


    print()

def demonstrate_learned_policy(agent: QLearningAgent, env: GridWorld):
    """Demonstrate the learned policy of the agent."""
    state = env.reset()
    done = False
    total_reward = 0
    
    print("Demonstrating learned policy:")
    while not done:
        env.render()
        action = agent.get_action(state)  # Using epsilon = 0 would be better for pure exploitation
        state, reward, done = env.step(action)
        total_reward += reward
        time.sleep(0.5)  # Add delay to make visualization readable
        print("\033[H\033[J", end="")  # Clear screen
    
    env.render()  # Show final state
    print(f"Demonstration finished with total reward: {total_reward:.2f}")

def visualize_q_table(agent: QLearningAgent, env: GridWorld):
    """
    Visualize the Q-table showing the best action and its value for each state.
    Arrow symbols: ←↑→↓
    """
    action_symbols = ['→', '↓', '←', '↑']
    print("\nQ-table Visualization:")
    print("For each cell showing: Best Action (Q-value)")
    print("═" * (env.size * 12 + 1))
    
    for row in range(env.size):
        print("║", end=" ")
        for col in range(env.size):
            state = row * env.size + col
            best_action = np.argmax(agent.q_table[state])
            best_value = agent.q_table[state][best_action]
            
            # Format cell content
            cell = f"{action_symbols[best_action]}({best_value:.2f})"
            print(f"{cell:8}", end=" ║ ")
        print("\n" + "═" * (env.size * 12 + 1))

def visualize_state_values(agent: QLearningAgent, env: GridWorld):
    """
    Visualize the maximum Q-value for each state.
    This shows how valuable the agent thinks each position is.
    """
    print("\nState Values Visualization:")
    print("Shows max Q-value for each state")
    print("═" * (env.size * 8 + 1))
    
    for row in range(env.size):
        print("║", end=" ")
        for col in range(env.size):
            state = row * env.size + col
            state_value = np.max(agent.q_table[state])
            print(f"{state_value:6.2f}", end=" ║ ")
        print("\n" + "═" * (env.size * 8 + 1))

def visualize_policy_arrows(agent: QLearningAgent, env: GridWorld):
    """
    Visualize the learned policy using arrows.
    Shows just the direction the agent would take in each state.
    """
    action_symbols = ['→', '↓', '←', '↑']
    print("\nLearned Policy Visualization:")
    print("═" * (env.size * 4 + 1))
    
    for row in range(env.size):
        print("║", end=" ")
        for col in range(env.size):
            state = row * env.size + col
            if state == env.goal:
                print("G", end=" ║ ")
            else:
                best_action = np.argmax(agent.q_table[state])
                print(f"{action_symbols[best_action]}", end=" ║ ")
        print("\n" + "═" * (env.size * 4 + 1))

if __name__ == "__main__":
    agent = QLearningAgent(BOARDSIZE ** 2, 4)

    rewards = train_agent(agent, episodes=300, render=False)
    
    # Create a new environment and agent to demonstrate learned policy
    env = GridWorld(BOARDSIZE, OBSTACLES)
    demonstrate_learned_policy(agent, env)
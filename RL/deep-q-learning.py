import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

from main import GridWorld

OBSTACLES = [3,24,35,55,56,57,58, 59, 82, 92]
DISCOUNT_FACTOR = 0.95


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output a single value
        return x

def train(render=False):
    epsilon = 1
    epsilon_decay = 0.998
    min_epsilon = 0.05
    env = GridWorld(10, OBSTACLES)
    critic = Critic(104)

    rewards_history = []

    optimizer_critic = optim.Adam(critic.parameters(), lr=0.01)
    for epoch in range(1000):
        env.reset()
        state = env.reset()
        total_reward = 0
        done = False

        for move in range(300):
            if render:
                env.render()
                #input()  # Add delay to make visualization readable
                print("\033[H\033[J", end="")  # Clear screen (works in most terminals)
            
            action = action_by_best_score(epsilon, env, critic, state)
            # actor_output = actor(encodestate(state, env.size))
            # action = determine_action(actor_output, epsilon)
            score = critic(critic_input(state, action, env.size)).detach()

            next_state, reward, done = env.step(action)

            
           
            next_action = action_by_best_score(0, env, critic, next_state)
            score = critic(critic_input(state, action, env.size))
            score_next_step = critic(critic_input(next_state, next_action, env.size)).detach()
            if not done:
                critic_loss = (score - (score_next_step * DISCOUNT_FACTOR + reward)) ** 2
            else:
                critic_loss = (score - reward)**2
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()
            
            state = next_state
            action = next_action
            total_reward += reward

            if done:
                break
            
            
        if render:
            env.render()  # Show final state
            print(f"Episode {epoch + 1} finished with reward: {total_reward:.2f}")
            time.sleep(1)
            
        rewards_history.append(total_reward)

        epsilon = max(epsilon_decay * epsilon, min_epsilon)
        
        if (epoch + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {epoch + 1}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")
        
        

    input()
    demonstrate_learned_policy(critic=critic, env=env)

def action_by_best_score(epsilon, env, critic, state):
    if random.random() < epsilon:
        action = random.randint(0,3)
    else:
        action = np.argmax([critic(critic_input(state, action, env.size)).detach().numpy() for action in range(4)])
    return action

def determine_action(actor_output, epsilon):
    if random.random() < epsilon:
        return random.randint(0,5) % 4
    return random.choices(range(len(actor_output)), weights=actor_output)[0]

def encodestate(state, gridsize):
    pos = F.one_hot(torch.tensor(state), gridsize ** 2).float()
    return pos

def critic_input(state, action, gridsize):
    return torch.concat((encodestate(state, gridsize), F.one_hot(torch.tensor(action), 4)))

def demonstrate_learned_policy(critic, env: GridWorld):
    """Demonstrate the learned policy of the agent."""
    state = env.reset()
    done = False
    total_reward = 0
    
    print("Demonstrating learned policy:")
    while not done:
        env.render()
        action = action_by_best_score(0.02, env, critic, env.state)  # Using epsilon = 0 would be better for pure exploitation
        state, reward, done = env.step(action)
        total_reward += reward
        time.sleep(0.5)  # Add delay to make visualization readable
        print("\033[H\033[J", end="")  # Clear screen
    
    env.render()  # Show final state
    print(f"Demonstration finished with total reward: {total_reward:.2f}")

if __name__ == "__main__":
    train(render=False)

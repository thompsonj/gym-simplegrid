import os
import sys
import argparse
import gym
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

import gym_simplegrid
from utils import smooth, Timer
from networks import PolicyMLP, CriticMLP

class MLPAgent:
    """Actor-Critic Agent for A2C."""
    def __init__(self, n_actions, n_states, hidden_size, learning_rate, gamma=0.99):
        self.gamma = gamma
        self.n_actions = n_actions
        self.actor= PolicyMLP(n_actions,n_states, hidden_size['actor'])
        self.critic = CriticMLP(n_states, hidden_size['critic'])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate['actor'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate['critic'])
        self.history = []

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float32)
        probs = self.actor(state)
        action_probs = Categorical(probs = probs)
        action = action_probs.sample()

        return action.detach().numpy().item()

    def store_transition(self, observation, action, reward, next_observation):
        self.history.append((observation, action, reward, next_observation))
    
    def reset(self):
        self.history = []

    def learn(self, done):
        """Apply A2C updates."""
        current_state, current_action,current_reward, next_state = self.history[-1]
        current_state = torch.tensor(current_state, dtype = torch.float32)
        next_state = torch.tensor(next_state, dtype = torch.float32)
        current_reward = torch.tensor(current_reward, dtype = torch.float32)
        current_action = torch.tensor(current_action, dtype = torch.float32)
        
        actor_output = self.actor(current_state)
        action_probs = Categorical(probs = actor_output)
        
        current_state_value = self.critic(current_state)
        next_state_value = self.critic(next_state)

        # Calculate the losses
        td_error = (current_reward + self.gamma*next_state_value*(1- int(done))) - current_state_value
        log_probs = action_probs.log_prob(current_action)
        actor_loss = -td_error * log_probs
        critic_loss = td_error ** 2
        loss = actor_loss+critic_loss
        
        # Take gradient step
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item(), next_state_value.item()


def get_state_rep(observation, state_rep, n_states, ncols, nrows, env, goal_r=None, goal_c=None):
    """Prepare state representation to be passed on as input to actor-critic agent."""
    if state_rep == 'one-hot':
        # one hot representation of the location of the agent
        observation = nn.functional.one_hot(torch.tensor(observation), num_classes=n_states)
    elif state_rep == 'row-column':
        # two one-hot vectors concatenated, one indicating the row, the other the column of the agent's position
        col, row = (observation % ncols, observation // ncols)
        col_oh = nn.functional.one_hot(torch.tensor(col), num_classes=ncols)
        row_oh = nn.functional.one_hot(torch.tensor(row), num_classes=nrows)
        observation = torch.cat((col_oh, row_oh))
    elif state_rep == 'rgb':
        # RGB rendering of the environment
        state = env.render(mode='rgb_array')
        observation = state.flatten() / 255
    elif state_rep == 'map':
        col, row = (observation % ncols, observation // ncols)
        desc = env.desc.copy()
        desc[row, col] = b'A'
        letter_map = {b'E':0.0,  b'W':-1.0, b'A':1.0, b'B':0.5, b'X':0.25, b'G':0.75, b'S':0.0}
        observation = np.vectorize(letter_map.get)(desc)
        observation = observation.flatten()
    elif 'agent+goal row-col':
        col, row = (observation % ncols, observation // ncols)
        col_oh = nn.functional.one_hot(torch.tensor(col), num_classes=ncols)
        row_oh = nn.functional.one_hot(torch.tensor(row), num_classes=nrows)
        goal_col_oh = nn.functional.one_hot(torch.tensor(goal_c), num_classes=ncols)
        goal_row_oh = nn.functional.one_hot(torch.tensor(goal_r), num_classes=nrows)
        observation = torch.cat((col_oh, row_oh, goal_col_oh, goal_row_oh))
        
    return observation


def train_A2C(my_desc, my_reward_map, config):
    timer = Timer()
    # %% PARAMETERS
    n_steps = 8
    n_episodes = config.n_episodes
    gamma = 0.99  # discount factor
    hidden_size = {'actor':128, 'critic':128}
    learning_rate = {'actor': 1e-4, 'critic': 1e-4}
    n_states = sum([len(row) for row in my_desc])
    nrows = len(my_desc)
    ncols = len(my_desc[0])
    state_size = nrows + ncols
    # How large is the input to the network, depending on the state representation
    input_size = {'one-hot':n_states, 'row-column':state_size, 'rgb':n_states*48, 
                  'map':n_states, 'agent+goal row-col':state_size*2}
    
    # %% INITIALIZATIONS
    env = gym.make('SimpleGrid-v0', desc=my_desc, reward_map=my_reward_map, task=config.task)
    agent = MLPAgent(n_actions=env.action_space.n, n_states=input_size[config.state_rep], hidden_size=hidden_size, learning_rate=learning_rate, gamma=gamma)
    score_history = []
    actor_losses = []
    critic_losses = []
    action_map = {0: 'LEFT ', 1: 'DOWN ', 2: 'RIGHT', 3: "UP   "}
    agent.reset()
    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()
        observation = get_state_rep(observation, config.state_rep, n_states, ncols, nrows, env)
        mean_actor_loss_in_episode = 0
        mean_critic_loss_in_episode = 0
        step_count = 0
        while not done:
            action = agent.choose_action(observation)
            if config.render:
                env.render(mode="human")
            next_observation, reward, done, info = env.step(action)
            done = done or (step_count >= n_steps)
            next_observation = get_state_rep(next_observation, config.state_rep, n_states, ncols, nrows, env)
            agent.store_transition(observation, action, reward, next_observation)
            observation = next_observation
            score += reward
            actor_loss, critic_loss, value = agent.learn(done)
            print(f'Episode {i}, step {step_count}, action {action_map[action]}, reward {reward}, value {value:.2f}        ', end='\r')

            mean_actor_loss_in_episode += actor_loss
            mean_critic_loss_in_episode += critic_loss
            step_count += 1
            
        if config.render:
            env.render(mode="human")
        
        mean_actor_loss_in_episode = mean_actor_loss_in_episode / step_count
        mean_critic_loss_in_episode = mean_critic_loss_in_episode / step_count

        score_history.append(score)
        actor_losses.append(mean_actor_loss_in_episode)
        critic_losses.append(mean_critic_loss_in_episode)
        avg_score = np.mean(score_history[-50:])
        print('\n episode ', i, 'score  %.1f' % score, 'avg score %.1f' % avg_score)
    timer.stop_timer()
    
    # Plot learning curves
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all')
    ep, hist = smooth(score_history)
    ax1.plot(ep, hist)
    ax1.set_title('Score')
    ep, act_loss = smooth(actor_losses)
    ax2.plot(ep, act_loss)
    ax2.set_title('Actor Loss')
    ep, cri_losses = smooth(critic_losses)
    ax3.plot(ep, cri_losses)
    ax3.set_title('Critic Loss')
    ax3.set_xlabel('Episodes')
    plt.tight_layout()
    lr = learning_rate['actor']
    hsize = hidden_size['actor']
    filename = f'learning_curves_{config.task}_{config.state_rep}_h{hsize}_lr{lr}.png'
    plt.savefig(filename)
    
    
def meta_trainA2C(my_desc, my_reward_map, config):
    """ With the above as a starting point fill in this function according to the following specification:
    
    """
    pass


if __name__ == '__main__':
    """ Example use from command line:
    $  python3 actor_critic_solution.py --n_episodes=1000 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='find goal', help='find goal or collect')
    parser.add_argument('--state_rep', type=str, default='row-column', help='state representation. one of rgb, one-hot, row-column')
    parser.add_argument('--render', action='store_true', default=False, help='whether to render environment on each step')
    parser.add_argument('--n_episodes', type=int, default=1000)
    # parser.add_argument('--meta', action='store_true', default=False)
    config, _ = parser.parse_known_args()
    my_desc = [
        "WWWWWW",
        "WSEESW",
        "WESEEW",
        "WSEEGW",
        "WWWWWW"
    ]
    my_reward_map = {
        b'E': -0.1,
        b'S': -0.1,
        b'W': -1,
        b'G': 10.0,
    }

    train_A2C(my_desc, my_reward_map, config)
import os
import sys
import argparse
import random
from itertools import product
import gym
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

import gym_simplegrid
from utils import smooth, Timer, make_torch_float32
from networks import PolicyMLP, CriticMLP, PolicyRNN, CriticRNN

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
        state = make_torch_float32(observation)
        probs = self.actor(state)
        self.action_probs = Categorical(probs=probs)
        action = self.action_probs.sample()

        return action.detach().numpy().item()

    def store_transition(self, observation, action, reward, next_observation):
        self.history.append((observation, action, reward, next_observation))
    
    def reset(self):
        self.history = []

    def learn(self, done):
        """Apply A2C updates."""
        current_state, current_action,current_reward, next_state = self.history[-1]
        
        current_state = make_torch_float32(current_state)
        next_state = make_torch_float32(next_state)
        current_reward = make_torch_float32(current_reward)
        current_action = make_torch_float32(current_action)
        
        current_state_value = self.critic(current_state)
        next_state_value = self.critic(next_state)

        # Calculate the losses
        td_error = (current_reward + self.gamma*next_state_value*(1- int(done))) - current_state_value
        log_probs = self.action_probs.log_prob(current_action)
        actor_loss = -td_error * log_probs
        critic_loss = td_error ** 2
        loss = actor_loss+critic_loss
        
        # Take gradient step
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 2)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 2)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item(), next_state_value.item()

class RNNAgent:

    def __init__(self, n_actions, n_states, hidden_size, learning_rate, gamma=0.99):

        self.gamma = gamma
        self.n_actions = n_actions
        self.actor= PolicyRNN(n_actions, n_states, hidden_size['actor'])
        self.critic = CriticRNN(n_states, hidden_size['critic'])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate['actor'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate['critic'])
        self.history = []

    def reset(self):
        self.actor_hidden = None
        self.critic_hidden = None
        self.current_state_value = None
        self.history = []

    def choose_action(self, observation):
        ''' Selects an action by using the policy network. 

        Arguments
        ---------
        observation : np.array [1, n_features]

        Returns
        -------
        action : int between 0 and n_action - 1
        '''
        
        state = make_torch_float32(observation)
        log_probs, self.actor_hidden = self.actor(state, self.actor_hidden)
        self.action_probs = Categorical(logits=log_probs)
        action = self.action_probs.sample()
        return action.detach().numpy().item()

    def store_transition(self, observation, action, reward, next_observation):
        self.history.append((observation, action, reward, next_observation))

    def learn(self, done):
        current_state, current_action, current_reward, next_state = self.history[-1]
        current_state = make_torch_float32(current_state)
        next_state = make_torch_float32(next_state)
        current_reward = make_torch_float32(current_reward)
        current_action = make_torch_float32(current_action)
        
        if self.current_state_value is None:
            self.current_state_value, self.critic_hidden = self.critic(current_state, self.critic_hidden)
            # print(self.current_state_value)

        next_state_value, next_critic_hidden = self.critic(next_state, self.critic_hidden)

        # Calculate loss
        td_error = (current_reward + self.gamma*next_state_value*(1 - int(done))) - self.current_state_value
        log_probs = self.action_probs.log_prob(current_action)
        actor_loss = -td_error * log_probs
        critic_loss = td_error ** 2
        loss = actor_loss+critic_loss
        
        # Apply gradient update
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.current_state_value = next_state_value.detach()
        self.critic_hidden = next_critic_hidden

        self.actor_hidden = (self.actor_hidden[0].detach(), self.actor_hidden[1].detach())
        self.critic_hidden = (self.critic_hidden[0].detach(), self.critic_hidden[1].detach())

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
        # RGB rendering of the environment, a downsampled image of the world
        state = env.render(mode='rgb_array')
        observation = state.flatten() / 255
    elif state_rep == 'map':
        # Each world object is mapped to a distinct float value in an array with length = number of grid locations
        col, row = (observation % ncols, observation // ncols)
        desc = env.desc.copy()
        desc[row, col] = b'A'
        letter_map = {b'E':0.0,  b'W':-1.0, b'A':1.0, b'B':0.5, b'X':0.25, b'G':0.75, b'S':0.0}
        observation = np.vectorize(letter_map.get)(desc)
        observation = observation.flatten()
    elif 'agent+goal row-col':
        # 4-hot vector indicating row and column of both the agent and the goal location
        col, row = (observation % ncols, observation // ncols)
        col_oh = nn.functional.one_hot(torch.tensor(col), num_classes=ncols)
        row_oh = nn.functional.one_hot(torch.tensor(row), num_classes=nrows)
        goal_col_oh = nn.functional.one_hot(torch.tensor(goal_c), num_classes=ncols)
        goal_row_oh = nn.functional.one_hot(torch.tensor(goal_r), num_classes=nrows)
        observation = torch.cat((col_oh, row_oh, goal_col_oh, goal_row_oh))
        
    return observation


def get_env_dist(my_reward_map, task):
    """Returns a list of gym-simplegrid environments with different goal locations."""
    if task == 'find goal':
        base_desc = [
                "WWWWW",
                "WEEEW",
                "WEEEW",
                "WEEEW",
                "WWWWW"]
        nrows = len(base_desc)
        ncols = len(base_desc[0])
        
        candidate_goal_locs = [(row, col)  for row, col in product(range(1, nrows-1), range(1, ncols-1))]
        
        # fix start location
        start_row, start_col = (1, 1)
        candidate_goal_locs.remove((start_row, start_col))
        
        MDPs = []
        for row, col in candidate_goal_locs:
            temp = [list(string) for string in base_desc]
            temp[start_row][start_col] = 'S'
            temp[row][col] = 'G'
            grid = [''.join(lst) for lst in temp]
            env = gym.make('SimpleGrid-v0', desc=grid, reward_map=my_reward_map, task=task)
            MDPs.append(env)
    elif task == 'collect':
        base_desc = [
        "WWWWWW",
        "WSEEEW",
        "WEEEEW",
        "WEEEEW",
        "WEEEEW",
        "WWWWWW"]
        nrows = len(base_desc)
        ncols = len(base_desc[0])
        MDPs = []
        ball_locs_horiz = [[(row, 2), (row, 3), (row, 4)] for row in range(1, nrows-1)]
        ball_locs_vert = [[(2, col), (3, col), (4, col)] for col in range(1, ncols-1)]
        for ball_locs in ball_locs_horiz + ball_locs_vert:
            temp = [list(string) for string in base_desc]
            for row, col in ball_locs:
                temp[row][col] = 'B'
            grid = [''.join(lst) for lst in temp]
            # import  pdb;pdb.set_trace()
            env = gym.make('SimpleGrid-v0', desc=grid, reward_map=my_reward_map, task=task)
            MDPs.append(env)
                
        
    return MDPs, base_desc


def train_A2C(my_desc, my_reward_map, config):
    timer = Timer()
    # %% PARAMETERS
    n_steps = config.n_steps  #8
    n_episodes = config.n_episodes
    gamma = 0.99  # discount factor
    hidden_size = {'actor':32, 'critic':32}
    learning_rate = {'actor': config.lr, 'critic': config.lr}
    n_states = sum([len(row) for row in my_desc])
    nrows = len(my_desc)
    ncols = len(my_desc[0])
    state_size = nrows + ncols
    # How large is the input to the network, depending on the state representation
    input_size = {'one-hot':n_states, 'row-column':state_size, 'rgb':n_states*48, 
                  'map':n_states, 'agent+goal row-col':state_size*2}
    
    # %% INITIALIZATIONS
    env = gym.make('SimpleGrid-v0', desc=my_desc, reward_map=my_reward_map, task=config.task)
    if config.arch == 'mlp':
        agent = MLPAgent(n_actions=env.action_space.n, n_states=input_size[config.state_rep], hidden_size=hidden_size, learning_rate=learning_rate, gamma=gamma)
    elif config.arch == 'lstm':
        agent = RNNAgent(n_actions=env.action_space.n, n_states=input_size[config.state_rep], hidden_size=hidden_size, learning_rate=learning_rate, gamma=gamma)

    score_history = []
    actor_losses = []
    critic_losses = []
    action_map = {0: 'LEFT ', 1: 'DOWN ', 2: 'RIGHT', 3: "UP   "}
    for i in range(n_episodes):
        agent.reset()
        done = False
        score = 0
        observation = env.reset()
        observation = get_state_rep(observation, config.state_rep, n_states, ncols, nrows, env)
        mean_actor_loss_in_episode = 0
        mean_critic_loss_in_episode = 0
        step_count = 0
        while not done:
            action = agent.choose_action(observation)
            if config.render and not i%50:
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
            
        if config.render and not i % 50:
            env.render(mode="human")
        
        mean_actor_loss_in_episode = mean_actor_loss_in_episode / step_count
        mean_critic_loss_in_episode = mean_critic_loss_in_episode / step_count

        score_history.append(score)
        actor_losses.append(mean_actor_loss_in_episode)
        critic_losses.append(mean_critic_loss_in_episode)
        avg_score = np.mean(score_history[-50:])
        if not i%50:
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
    filename = f'learning_curves_{config.task}_{config.state_rep}_{config.arch}_h{hsize}_lr{lr}.png'
    plt.savefig(filename)
    
    
def meta_train_A2C(my_reward_map, config):
    """ Learn to learn in the recurrent dynamics.
    
    """
    timer = Timer()
    # %% PARAMETERS
    n_steps = config.n_steps  #8
    n_episodes = config.n_episodes
    gamma = 0.99  # discount factor
    hidden_size = {'actor':32, 'critic':32}
    learning_rate = {'actor': config.lr, 'critic': config.lr}
    
    # %% INITIALIZATIONS
    MDPs, base_desc = get_env_dist(my_reward_map, config.task)
    n_states = sum([len(row) for row in base_desc])
    nrows = len(base_desc)
    ncols = len(base_desc[0])
    state_size = nrows + ncols
    # How large is the input to the network, depending on the state representation
    input_size = {'one-hot':n_states, 'row-column':state_size, 'rgb':n_states*48, 
                  'map':n_states, 'agent+goal row-col':state_size*2}
    env = random.choice(MDPs)
    in_size = input_size[config.state_rep] + 1
    if config.arch == 'mlp':
        agent = MLPAgent(n_actions=env.action_space.n, n_states=in_size, hidden_size=hidden_size, learning_rate=learning_rate, gamma=gamma)
    elif config.arch == 'lstm':
        agent = RNNAgent(n_actions=env.action_space.n, n_states=in_size, hidden_size=hidden_size, learning_rate=learning_rate, gamma=gamma)

    score_history = []
    actor_losses = []
    critic_losses = []
    action_map = {0: 'LEFT ', 1: 'DOWN ', 2: 'RIGHT', 3: "UP   "}
    goal_row, goal_col = None, None
    agent.reset()
    for i in range(n_episodes):
        first_step = torch.ones(1, )
        done = False
        score = 0
        if not i % 5 or config.arch=='mlp':
            # print('### new goal location ###')
            env.close()
            # plt.ion()
            # plt.clf()
            # plt.close('all')
            render = False
            agent.reset()
            env = random.choice(MDPs)
            goal_row, goal_col = env.goal_loc
            if not i%1000 and config.render: # Every 1k episodes, render 5 episodes of metalearning
                render = True
            
        observation = env.reset()
        observation = get_state_rep(observation, config.state_rep, n_states, ncols, nrows, env, goal_row, goal_col)
        observation = torch.cat((first_step, observation))
        
        mean_actor_loss_in_episode = 0
        mean_critic_loss_in_episode = 0
        step_count = 0
        while not done:
            action = agent.choose_action(observation)
            first_step = torch.zeros(1,)
            if render:
                env.render(mode="human")
            next_observation, reward, done, info = env.step(action)
            done = done or (step_count >= n_steps)
            next_observation = get_state_rep(next_observation, config.state_rep, n_states, ncols, nrows, env, goal_row, goal_col)
            next_observation = torch.cat((first_step, next_observation))
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
        # Take avg score only for the last episode in each block
        avg_score = np.mean(score_history[::5][-50:]) if len(score_history) > 5 else 0
        if not i%50:
            print('\n episode ', i, 'score  %.1f' % score, 'avg score (at end of block) %.1f' % avg_score)
    timer.stop_timer()
        
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ep, hist = smooth(score_history[::5]) 
    ax1.plot(ep, hist)
    ax1.set_title('Score at the end of each block of 5 episodes')
    ax1.set_xlabel('Block')
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
    filename = f'metalearning_curves_{config.task}_{config.state_rep}_{config.arch}_h{hsize}_lr{lr}.png'
    plt.savefig(filename)


if __name__ == '__main__':
    """ Example use from command line:
    $  python3 actor_critic_solution.py --n_episodes=1000 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='find goal', help='find goal or collect')
    parser.add_argument('--state_rep', type=str, default='row-column', help='state representation. one of rgb, one-hot, row-column or agent+goal row-col')
    parser.add_argument('--render', action='store_true', default=False, help='whether to render environment on each step')
    parser.add_argument('--n_episodes', type=int, default=1000, help='how many episodes to train for')
    parser.add_argument('--n_steps', type=int, default=8, help='maximum number of steps to take within one episode')
    parser.add_argument('--arch', type=str, default='mlp', help='which model architecture to use. mlp or lstm')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--meta', action='store_true', default=False, help='whether to train on distribution of environments')
    config, _ = parser.parse_known_args()
    print(config)
    if config.task == 'find goal':
        my_desc = [
            "WWWWWWW",
            "WSEEESW",
            "WEESEEW",
            "WSEEEGW",
            "WWWWWWW"
        ]
    elif config.task == 'collect':
        my_desc = [
            "WWWWWWW",
            "WSEBBEW",
            "WEBEBEW",
            "WEEBEBW",
            "WWWWWWW"
        ]
    # Reward Function
    my_reward_map = {
        b'E': -0.1,
        b'S': -0.1,
        b'W': -2.0,
        b'G': 20.0,
        b'B': 10,
        # b'X': 5
    }
    if config.meta:
        meta_train_A2C(my_reward_map, config)
    else:
        train_A2C(my_desc, my_reward_map, config)
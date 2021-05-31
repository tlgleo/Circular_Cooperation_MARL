import gym
import circular_collect

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import numpy as np
import random
from collections import namedtuple, deque
import time

from agents_RL import *
from utils import *

env = gym.make('circular_collect_special_4x4-v0')
states = env.reset()
nb_agents = 4
t_sleep = 2
t_max = 100
several_agents = False

agent_generic = get_agent(80, 5, model_name= 'Q_gen_P_a_for_x.pth', agent_name = 'agent_gen')



def preprocess_obs(state):
    state = preprocess_state(state, 11, 11, 0, 0)
    state = preprocess_state_layers(state, 0, []).reshape(1,-1)
    (_,n) = np.shape(state)
    state = 1.0 * state + np.random.rand(1,n)/100.0
    return state



if not several_agents:
    for _ in range(t_max):
        state = states[0]
        state = preprocess_obs(state)
        a0 = agent_generic.act(state, eps=0.0)
        ac = [a0, 4,4,4]
        states, r, done, _ = env.step(ac)
        env.render(mode='human', highlight=False, save_fig=False)
        print(r)
        time.sleep(t_sleep)
else:
    for _ in range(t_max):
        actions = []
        for i in range(nb_agents):
            state = states[i]
            state = preprocess_obs(state)
            ai = agent_generic.act(state, eps=0.0)
            actions.append(ai)
        states, r, done, _ = env.step(actions)
        env.render(mode='human', highlight=False, save_fig=False)
        print(r)
        time.sleep(t_sleep)

def dqn(n_episodes=2000, state_size = 80, action_size = 5, max_t=100, eps_start=0.3, eps_end=0.01, eps_decay=0.9):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    agent = get_agent(state_size=state_size, action_size=action_size)


    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        states = env.reset()
        print(np.shape(states[0]))
        score = 0
        for t in range(max_t):
            state = preprocess_obs(states[0])
            action = agent.act(state, eps)

            actions = [action, 4, 4, 4]
            next_states, rewards, dones, _ = env.step(actions)

            next_state = preprocess_obs(next_states[0])
            reward = rewards[0]
            done = False

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        #eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 10 == 0:
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint' + '.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'Q_gen_P_a_for_x.pth')
            break
    return scores



#scores = dqn()
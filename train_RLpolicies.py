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


state_size = 16
action_size = 5

env = gym.make('circular_collect_special_4x4-v0')
s = env.reset()
print(np.shape(s))

print(env.n_agents)



def learn_for_agent_discrete(env, state_size, action_size, ident_agent, partition, path_model):
    agent = get_agent(state_size=state_size, action_size=action_size, model_name=None)
    model_name = create_name_model(ident_agent, discrete_coop=True, partition = partition)
    path_model = path_model+model_name
    torch.save(agent.qnetwork_local.state_dict(), path_model)


def learn_for_agent_continuous(env, state_size, action_size, ident_agent, path_model):
    agent = get_agent(state_size=state_size, action_size=action_size, model_name=None)
    model_name = create_name_model(ident_agent, discrete_coop=False)
    path_model = path_model+model_name
    torch.save(agent.qnetwork_local.state_dict(), path_model)


#learn_for_agent_discrete(env, 128, 5, 0, '12', path)


def learn_all_models_discrete_selfplay(env, state_size, action_size, path_model):
    for ident_agent in range(env.n_agents):
        partition_set_agent = create_partition_set_agent(env.n_agents, ident_agent)
        for part in partition_set_agent:
            learn_for_agent_discrete(env, state_size, action_size, ident_agent, part, path_model)

def learn_all_models_continous_selfplay(env, state_size, action_size, path_model):
    for ident_agent in range(env.n_agents):
        learn_for_agent_continuous(env, state_size, action_size, ident_agent, path_model)



def learn_all_models_discrete_generic_selfplay(env, state_size, action_size, path_model):
    # in a symmetrical environment, for the proof of concept, we train only some generic policies
    # for computational reduction
    n_agents = env.n_agents
    list_model_names = name_generic_model_list(n_agents)
    index_agents_rewards = [0]*n_agents

    for i,model_name in enumerate(list_model_names):

        # create a fixed fonction to preprocess the list of rewards from the agents
        index_agents_rewards[i] = 1
        def process_rewards(list_rewards_agents):
            return transform_reward(0,list_rewards_agents, index_agents=index_agents_rewards, normalise=1.0)


        agent = get_agent(state_size=state_size, action_size=action_size, model_name=None)
        path_name = path_model + model_name
        torch.save(agent.qnetwork_local.state_dict(), path_name)



def learn_discrete_coop_stationnary(env, state_size, action_size, path_policy_model, coop_vector = [1,0,0,0]):




path = './models/2/'

learn_all_models_discrete_generic_selfplay(env, 80, 5, path)



#learn_all_models_discrete_selfplay(env, 128, 5, path)
#learn_all_models_continous_selfplay(env, 132, 5, path)



model_name = path + 'Q_gen_P_a_for_bc.pth'

#model_name_continuous = path+create_name_model(0, discrete_coop=False)


agent = get_agent(state_size=80, action_size=5, model_name=model_name)

s = preprocess_state(s[0],11,11,0,1)

s = preprocess_state_layers(s, 0, [1,2])

print(np.shape(s))
s = s.flatten()

print(agent.act(s))



env.render(mode='human', highlight=False)
time.sleep(100)
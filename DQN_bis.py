import gym
import circular_collect

import numpy as np
import torch
#from IPython.display import clear_output
import random
from matplotlib import pylab as plt
import copy
from collections import deque

import time

from agents_RL import *
from utils import *



class Agent_DQN:
    def __init__(self, state_size = 80, action_size = 5,
                 layer_size_1 = 150, layer_size_2 = 100,
                 model_weights = None):
        self.state_size = state_size
        self.action_size = action_size
        self.model = torch.nn.Sequential(
                        torch.nn.Linear(state_size, layer_size_1),
                        torch.nn.ReLU(),
                        torch.nn.Linear(layer_size_1, layer_size_2),
                        torch.nn.ReLU(),
                        torch.nn.Linear(layer_size_2, action_size)
                        )

        if model_weights is not None:
            self.model.load_state_dict(torch.load(model_weights))

    def q_values(self, state):
        return self.model(state).data.numpy()

    def act(self, state):
        q_val = self.q_values(state)
        return np.argmax(q_val)

#model_path = 'Q_gen_P_a_for_bc.pth'
model_path = 'POLICY_QUI_MARCHE'
agent_commun = Agent_DQN(80,5,150,100,model_path)

l1 = 80
l2 = 150
l3 = 100
l4 = 5

s = np.zeros([80,1]).reshape(1,-1)
s1 = torch.from_numpy(s).float()
print("agent Q values", agent_commun.q_values(s1))

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3,l4)
)

model2 = copy.deepcopy(model) #A
model2.load_state_dict(model.state_dict()) #B

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

gamma = 0.9
epsilon = 0.3

#epochs = 0
losses = []
mem_size = 3000
batch_size = 200
replay = deque(maxlen=mem_size)
max_moves = 30
h = 0
#sync_freq = 500  # A
sync_freq = 100
j = 0

test = True
t_sleep = 0.1

if test:
    epochs = 0
    t_max = 1000
    env_name = "circular_collect_special_4x4-v2"
else:
    epochs = 10000
    t_max = 0
    max_moves = 50
    env_name = "circular_collect_special_4x4_tr-v0"


env = gym.make(env_name)
env.reset(seed = 201)
states = env.reset()

def preprocess_obs(state, i, j):
    IDENT_TO_COORD = [(0, 0), (1, 0), (1, 1), (0, 1)]
    (k,l) = IDENT_TO_COORD[i]
    state = preprocess_state(state, 11, 11, k, l)
    #state = preprocess_state_layers_for_agent(state, i, [j]).reshape(1,-1)
    state = state[:,:,:5]
    state = state.reshape(1,-1)
    (_,n) = np.shape(state)
    state = 1.0 * state + np.random.rand(1,n)/100.0
    return state

if test:
    print("load")
    model_weights = 'checkpoint_A-B.pth'
    model.load_state_dict(torch.load(model_weights))



delta = 0
path = 'render/4/'
for t in range(t_max):
    actions = []
    for i in [0,1,2,3]:
        state = states[i]
        state = preprocess_obs(state, i, (i+delta)%4)
        print(np.shape(state))
        state1 = torch.from_numpy(state).float()
        #qval = model(state1)
        #qval_ = qval.data.numpy()
        a = agent_commun.act(state1)
        #a = np.argmax(qval_)
        qval_ = agent_commun.q_values(state1)
        print('step ', t)
        print(qval_)
        actions.append(a)



    env.render(mode='human', highlight=False, save_fig=True, fig_name = path+str(t)+'.png')

    #a0 = np.argmax(qval_)
    #ac = [4, 4, 4, a0]
    states, r, done, _ = env.step(actions)

    print('rewards = ',r)

    time.sleep(t_sleep)




for i in range(epochs):
    env.reset(seed = i+1)
    states = env.reset(seed = i+1)
    state1_ = preprocess_obs(states[0],0,0)
    state1 = torch.from_numpy(state1_).float()
    status = 1
    mov = 0
    while (status == 1):
        j += 1
        mov += 1
        qval = model(state1)
        qval_ = qval.data.numpy()
        if (random.random() < epsilon):
            action = np.random.randint(0, 5)
        else:
            action = np.argmax(qval_)

        actions = [action, 4, 4, 4]
        next_states, rewards, dones, _ = env.step(actions)
        #print(rewards)
        state2_ = preprocess_obs(next_states[0],0,0)
        state2 = torch.from_numpy(state2_).float()
        #reward = rewards[0] + 10*rewards[1]
        reward = rewards[0] + rewards[1]
        #print(reward)
        done = False
        exp = (state1, action, reward, state2, done)
        replay.append(exp)  # H
        state1 = state2

        if len(replay) > batch_size:
            minibatch = random.sample(replay, batch_size)
            state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
            action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
            reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
            done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])
            Q1 = model(state1_batch)
            with torch.no_grad():
                Q2 = model2(state2_batch)  # B

            Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X, Y.detach())
            print(i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            if j % sync_freq == 0:  # C
                model2.load_state_dict(model.state_dict())
                torch.save(model.state_dict(), 'checkpoint_A-B' + '.pth')
        if mov > max_moves:
            status = 0
            mov = 0

losses = np.array(losses)
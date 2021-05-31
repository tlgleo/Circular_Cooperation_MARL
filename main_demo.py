import gym
import circular_collect

import numpy as np
import os
from display_functions import *
from agent_rlTFT import *
from run_game import run_game
from social_metrics import *
from agents_RL import *

name_env = "circular_collect_special_4x4-v0"

env = gym.make(name_env)
env.reset(seed = 1234)

ident = 1
agent_grTFT = []
state_size = 128
action_size = 5


CIRC_PROBAS_4P_SC = \
    [[0.25, 0.50, 0.25, 0.00],
     [0.00, 0.25, 0.50, 0.25],
     [0.25, 0.00, 0.25, 0.50],
     [0.50, 0.25, 0.00, 0.25]
    ]

CIRC_PROBAS_4P_P = \
    [[0.5, 0.2, 0.2, 0.1],
     [0.1, 0.5, 0.2, 0.2],
     [0.2, 0.1, 0.5, 0.2],
     [0.2, 0.2, 0.1, 0.5]
    ]

#without cycles
CIRC_PROBAS_4P_0C = \
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0],
    ]

CIRC_PROBAS_4P_1C = \
    [[0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0, 0.0],
    ]

CIRC_PROBAS_4P_2C = \
    [[0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
    ]

CIRC_PROBAS_4P_B = \
    [[0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
    ]



double_matrix = np.array([[0,1,1,0],[0,0,1,1],[1,0,0,1],[1,1,0,0]])*1.0

#max_coop_graph_env = env.prob_matrix
max_coop_graph_env = double_matrix
print(max_coop_graph_env)
print(np.shape(max_coop_graph_env))


agent_grTFT0 = Agent_TFT_Gamma(0, 'agent0', n_agents=4, max_coop_matrix=max_coop_graph_env)
agent_grTFT1 = Agent_TFT_Gamma(1, 'agent1', n_agents=4, max_coop_matrix=max_coop_graph_env)
agent_grTFT2 = Agent_TFT_Gamma(2, 'agent2', n_agents=4, max_coop_matrix=max_coop_graph_env)
agent_grTFT3 = Agent_TFT_Gamma(3, 'agent3', n_agents=4, max_coop_matrix=max_coop_graph_env)
traitor = Agent_Traitor(3,'traitor', max_coop_matrix=max_coop_graph_env, t_traitor=[30,60])
nice = Agent_Nice(3,'nice', max_coop_matrix=max_coop_graph_env)
egoist = Agent_Egoist(3,'egoist', max_coop_matrix=max_coop_graph_env)


# CREATE n_agent Nice "TFT agents"
nices_algos = [Agent_Nice(i, 'N', n_agents=4, max_coop_matrix=max_coop_graph_env) for i in range(4)]

# CREATE n_agent Egoist "TFT agents"
egoists_algos = [Agent_Egoist(i, 'D', n_agents=4, max_coop_matrix=max_coop_graph_env) for i in range(4)]


list_algos_tft = [agent_grTFT0, agent_grTFT1, agent_grTFT2, agent_grTFT3]

agentRL = Agent_DQN(state_size=80, action_size=5, model_weights='checkpoint.pth')

agentRLTFT0 = Agent_RLTFT(0, agent_grTFT0, agentRL, state_size, action_size, n_agents=4, env_state_size = (11,11))
agentRLTFT1 = Agent_RLTFT(1, agent_grTFT1, agentRL, state_size, action_size, n_agents=4, env_state_size = (11,11))
agentRLTFT2 = Agent_RLTFT(2, agent_grTFT2, agentRL, state_size, action_size, n_agents=4, env_state_size = (11,11))
agentRLTFT3 = Agent_RLTFT(3, traitor, agentRL, state_size, action_size, n_agents=4, env_state_size = (11,11))

l_agents = [agentRLTFT0, agentRLTFT1, agentRLTFT2, agentRLTFT3]

print("parameters TFT", l_agents[0].agent_grTFT.parameters_TFT)
print("matrix max ", l_agents[0].agent_grTFT.max_coop_matrix)

coop_max_matrix = np.ones([4,4])
current_payoffs = [-1,3,1,4]
output_name = 'graph.png'

c1 = [1]*20
c2 = [2]*20
c3 = [3]*20
c4 = [4]*20
curves = [c1,c2,c3,c4]





#l_agents[0].agent_grTFT = nices_algos[0]

env.reset(seed = 12345)
curves = run_game(env, list_agents= l_agents, t_max=200, k_smooth=30, limits_y=(-10, 50), render_env = True, render = True, name_expe = '0000_detect_coop3_traitor', steps_change_max_coop=[205, 205], max_coop_matrix_change=[CIRC_PROBAS_4P_2C, CIRC_PROBAS_4P_B])

print('values of sum')
s = 0
for c in curves:
    print(c[-1])
    s += c[-1]
print('sum', s)
#compute_metrics(env, l_agents, list_algos_tft, 100)
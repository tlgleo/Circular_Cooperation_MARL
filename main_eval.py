import gym
import circular_collect

import numpy as np
import os
from display_functions import *
from agent_rlTFT import *
from run_game import run_game
from social_metrics import *
from agents_RL import *
import statistics as st
import pickle
from scipy.spatial.distance import euclidean

name_env = "circular_collect_special_4x4-v0"

env = gym.make(name_env)
env.reset(seed = 103)

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

agent_grTFT0 = Agent_TFT_Gamma(0, 'agent0', n_agents=4, max_coop_matrix=max_coop_graph_env)
agent_grTFT1 = Agent_TFT_Gamma(1, 'agent1', n_agents=4, max_coop_matrix=max_coop_graph_env)
agent_grTFT2 = Agent_TFT_Gamma(2, 'agent2', n_agents=4, max_coop_matrix=max_coop_graph_env)
agent_grTFT3 = Agent_TFT_Gamma(3, 'agent3', n_agents=4, max_coop_matrix=max_coop_graph_env)

agent_TFT0 = Agent_TFT_NoGraph_Gamma(0, 'agent0', n_agents=4, max_coop_matrix=max_coop_graph_env)
agent_TFT1 = Agent_TFT_NoGraph_Gamma(1, 'agent1', n_agents=4, max_coop_matrix=max_coop_graph_env)
agent_TFT2 = Agent_TFT_NoGraph_Gamma(2, 'agent2', n_agents=4, max_coop_matrix=max_coop_graph_env)
agent_TFT3 = Agent_TFT_NoGraph_Gamma(3, 'agent3', n_agents=4, max_coop_matrix=max_coop_graph_env)

traitor = Agent_Traitor(3,'traitor', max_coop_matrix=max_coop_graph_env, t_traitor=[50,60])

list_algos_tft = [agent_grTFT0, agent_grTFT1, agent_grTFT2, agent_grTFT3]

list_algos_tft_classic = [agent_TFT0, agent_TFT1, agent_TFT2, agent_TFT3]

agentRL = Agent_DQN(state_size=80, action_size=5, model_weights='checkpoint.pth')

agentRLTFT0 = Agent_RLTFT(0, agent_grTFT0, agentRL, state_size, action_size, n_agents=4, env_state_size = (11,11))
agentRLTFT1 = Agent_RLTFT(1, agent_grTFT1, agentRL, state_size, action_size, n_agents=4, env_state_size = (11,11))
agentRLTFT2 = Agent_RLTFT(2, agent_grTFT2, agentRL, state_size, action_size, n_agents=4, env_state_size = (11,11))
agentRLTFT3 = Agent_RLTFT(3, agent_grTFT3, agentRL, state_size, action_size, n_agents=4, env_state_size = (11,11))

# CREATE n_agent Nice "TFT agents"
nices_algos = [Agent_Nice(i, 'N', n_agents=4, max_coop_matrix=max_coop_graph_env) for i in range(4)]

# CREATE n_agent Egoist "TFT agents"
egoists_algos = [Agent_Egoist(i, 'D', n_agents=4, max_coop_matrix=max_coop_graph_env) for i in range(4)]

l_agents = [agentRLTFT0, agentRLTFT1, agentRLTFT2, agentRLTFT3]

def mean_std(values, rd = 3):
    m = st.mean(values)
    std = st.stdev(values)
    return (np.round(m, rd) , np.round(std, rd))


def efficiency(curves, rd = 3):
    n = len(curves)
    t_max = len(curves[0])
    u = 0
    for c in curves:
        u += c[-1]
    output = u/t_max
    return np.round(output, decimals=rd)

def safety(curves_agent_vs_allegoists, curves_all_egoists, rd = 3):
    n = len(curves_agent_vs_allegoists)
    t_max = len(curves_agent_vs_allegoists[0])
    r1 = curves_agent_vs_allegoists[0][-1] # agent face to all egoists
    r2 = curves_all_egoists[0][-1]  # all egoist
    for i in range(0):
        pass
        #u1 += curves_agent_vs_allegoists[i][-1]
        #u2 += curves_all_egoists[i][-1]
    #print("safety ", r1, r2)
    output = (r1 - r2)/t_max
    return np.round(output, decimals=rd)


def incentive_compatibility(curves_1cooperator_vs_agents, curves_1defector_vs_agents, rd = 3):
    n = len(curves_1cooperator_vs_agents)
    t_max = len(curves_1cooperator_vs_agents[0])
    r1 = curves_1cooperator_vs_agents[0][-1] # payoff if cooperation
    r2 = curves_1defector_vs_agents[0][-1] # payoff if defection
    #print(r1, r2)
    output = (r1 - r2)/t_max
    return np.round(output, decimals=rd)



def compute_mean_social_metrics(env, list_agents,
                                list_TFT, egoists_algos = egoists_algos, nices_algos = nices_algos,
                                n_runs = 5 , t_max = 200, choose_agent_for_incent = False,
                                given_potential = True, given_dectection = True,
                                change_coop_graph = [],
                                change_t_moments = [],
                                ):

    n_agents = len(list_agents)

    for i in range(n_agents):
        list_agents[i].agent_grTFT = list_TFT[i]

    efficiencies = []
    incentives = []
    safeties = []

    # for efficiency:
    for k in range(n_runs):
        env.reset(seed=100 + k)
        curves_all_agents = run_game(env, list_agents=l_agents, t_max=t_max, render=False, name_expe='00TestPotenew', given_detection=given_dectection,
given_potential=given_potential, max_coop_matrix_change=change_coop_graph, steps_change_max_coop=change_t_moments)


        #curves_all_agents = run_game(env, list_agents=l_agents, t_max=t_max, render=True, render_env=True, name_expe = '0_TFT_6', limits_y=(-10,100), steps_change_max_coop=[10], max_coop_matrix_change=[CIRC_PROBAS_4P_B])
        ef = efficiency(curves_all_agents)
        print('efficiency ', k, ef)
        efficiencies.append(ef)

    # for incentive-compatibility
    for k in range(0):
        if choose_agent_for_incent:
            list_agents[0].agent_grTFT = list_TFT[0]
        else:
            list_agents[0].agent_grTFT = nices_algos[0]
        env.reset(seed=200 + k)
        #curves_1cooperator_vs_agents = run_game(env, list_agents=l_agents, t_max=t_max, render=True, render_env=False, name_expe = '0_incentive1', limits_y=(-10,100))
        curves_1cooperator_vs_agents = run_game(env, list_agents=l_agents, t_max=t_max, render=False, given_detection=given_dectection, given_potential=given_potential,
                                                                                    max_coop_matrix_change=change_coop_graph, steps_change_max_coop=change_t_moments)
        list_agents[0].agent_grTFT = egoists_algos[0]
        env.reset(seed=300 + k)
        curves_1defector_vs_agents = run_game(env, list_agents=l_agents, t_max=t_max, render=False, given_detection=given_dectection, given_potential=given_potential,
                                                                                    max_coop_matrix_change=change_coop_graph, steps_change_max_coop=change_t_moments)
        ic = incentive_compatibility(curves_1cooperator_vs_agents, curves_1defector_vs_agents)
        print('incentive ', k, ic)
        incentives.append(ic)

    # for safety:
    for k in range(0):

        # all egoists
        for i in range(n_agents):
            list_agents[i].agent_grTFT = egoists_algos[i]
        env.reset(seed=400 + k)
        curves_all_egoists = run_game(env, list_agents=l_agents, t_max=t_max, render=False, given_detection=given_dectection, given_potential=given_potential,
                                                                                    max_coop_matrix_change=change_coop_graph, steps_change_max_coop=change_t_moments)


        # 1 agent vs (n-1) egoist
        list_agents[0].agent_grTFT = list_TFT[0]
        env.reset(seed=500 + k)
        curves_agent_vs_allegoists = run_game(env, list_agents=l_agents, t_max=t_max, render=False, given_detection=given_dectection, given_potential=given_potential,
                                                                                    max_coop_matrix_change=change_coop_graph, steps_change_max_coop=change_t_moments)
        #curves_agent_vs_allegoists = run_game(env, list_agents=l_agents, t_max=t_max, render=True, render_env=False, name_expe='0_safety_4', limits_y=(-10, 100))

        sf = safety(curves_agent_vs_allegoists, curves_all_egoists)
        print('safety ', k, sf)
        safeties.append(sf)


    #return [mean_std(efficiencies), mean_std(incentives), mean_std(safeties)]
    return [mean_std(efficiencies), mean_std([1,0]), mean_std([1,0])]

"""
print()
print("Bilateral ENV")
print()
print("TFT classic")
output = compute_mean_social_metrics(env, list_agents = l_agents, list_TFT = list_algos_tft_classic, n_runs = 5, t_max = 500)
print(output)
print()
"""
#change_t_moments = [250]
#change_coop_graph = [CIRC_PROBAS_4P_1C]
change_t_moments = []
change_coop_graph = []

#print("graph TFT")
output = compute_mean_social_metrics(env, list_agents = l_agents, list_TFT = egoists_algos, n_runs = 5, t_max = 500, change_coop_graph=change_coop_graph, change_t_moments=change_t_moments)
print(output)

#print("graph TFT - no Potential")
#output = compute_mean_social_metrics(env, list_agents = l_agents, list_TFT = list_algos_tft, n_runs = 3, t_max = 500, given_potential=False, change_coop_graph=change_coop_graph, change_t_moments=change_t_moments)
#print(output)
"""
print()
print("egoist")
output = compute_mean_social_metrics(env, list_agents = l_agents, list_TFT = egoists_algos, n_runs = 5, t_max = 500)
print(output)
print()
print("nices")
output = compute_mean_social_metrics(env, list_agents = l_agents, list_TFT = nices_algos, n_runs = 5, t_max = 500)
print(output)

"""
def compute_distance_graph(env, list_agents,
                                list_TFT,
                                n_runs = 5 ,
                                t_max = 200,
                                change_t_moments = None,
                                change_coop_graph = None
                           ):

    n_agents = len(list_agents)

    for i in range(n_agents):
        list_agents[i].agent_grTFT = list_TFT[i]

    curves_distances_current_coop = []
    curves_distances_potential = []


    for k in range(n_runs):
        env.reset(seed=100 + k)
        curves_payoffs, distances_current_coop, distances_potential_coop = run_game(env, list_agents=l_agents, t_max=t_max,
                                                                                    render=False, render_env=False, name_expe = '012_A',
                                                                                    dist_potential = True, dist_current_coop = False,
                                                                                    max_coop_matrix_change=change_coop_graph, steps_change_max_coop=change_t_moments)

        curves_distances_potential.append(distances_potential_coop)
        curves_distances_current_coop.append(distances_current_coop)

    return curves_distances_current_coop, curves_distances_potential

"""
curves1, curves2 = compute_distance_graph(env, l_agents,
                                list_algos_tft,
                                n_runs = 10 ,
                                t_max = 100,
                                change_t_moments = [50],
                                change_coop_graph = [CIRC_PROBAS_4P_B]
                           )

print(len(curves1))
print(len(curves1[0]))


f = open("distances_potential_5runs.obj","wb")
pickle.dump(curves2,f)
f.close()
"""
import numpy as np
import graphbasedTFT as grTFT
import torch_ac

Agent_TFT_Beta = grTFT.Agent_TFT_Beta
Agent_TFT_Gamma = grTFT.Agent_TFT_Gamma
max_coop_graph1 = np.array([[0,1,0.5,0],[0,0,1,0.5],[0.5,0,0,1],[1,0.5,0,0]])
print(max_coop_graph1)

print()
print()
print()
a0 = Agent_TFT_Beta(0, 'agent0', n_agents=4, max_coop_matrix=max_coop_graph1)

a0 = Agent_TFT_Gamma(0, 'agent1', n_agents=4, max_coop_matrix=max_coop_graph1)

coop_mat = np.zeros([4,4])
coop_mat[1,2] = 1.0
coop_mat[2,3] = 1.0
coop_mat[3,0] = 1.0

step_t = 0

for t in range(8):
    output = a0.act(coop_mat, t)
    print("current inner graph")
    print(a0.current_coop_matrix)
    print('output')
    print(output)
    print()

coop_mat = np.zeros([4, 4])
coop_mat[1, 2] = 0.0
coop_mat[2, 3] = 1.0
coop_mat[3, 0] = 1.0

print("TRAITOR")

for t in range(8):
    output = a0.act(coop_mat, t)
    print('inner source')
    print(a0.current_source)
    print("current inner graph")
    print(a0.current_coop_matrix)
    print('output')
    print(output)
    print()

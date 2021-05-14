import numpy as np
from utils import *
from agents_RL import *

IDENT_TO_COORD = [(0,0),(1,0),(0,1),(1,1)]

class Agent_RLTFT:
    """
    Agent mixing RL policies and graph-based Tit-for-Tat (TFT) strategies
    Only for execution phase
    """

    def __init__(self, ident, agent_grTFT, state_size, action_size, n_agents=4, env_state_size = (11,11)):
        self.ident = ident
        self.n_agents = n_agents
        self.agent_grTFT = agent_grTFT # agent dealing with graph-based cooperation (Tit-for-Tat)
        self.inner_coop_graph = np.eye(self.n_agents) # cooperation graph
        self.discrete_coop = True
        self.RL_policies_path = './'
        self.RL_policies = [[]]*self.n_agents # size = 2^(n_agents-1) if discrete_coop, size = 1 otherwise
        self.RL_generic_policies = [[]]*4
        self.max_coop_graph = np.eye(self.n_agents) # maximal possible cooperation graph (dynamic or fixed)
        self.state_size = state_size
        self.action_size = action_size
        self.env_state_size = env_state_size
        self.set_partition = create_partition_set(self.n_agents)

        self.k_steps_act_coop_max = 10  # act a fixed cooperation policy during k_steps_act_coop steps
        self.k_steps_act_coop = 0

        self.k_steps_dect_coop_max = 5  # for the (discrete) detection of cooperation
        self.k_steps_dect_coop_max = 0  # for the (discrete) detection of cooperation

        self.coop_receivers = [] # if discrete cooperation: fixed set of agents receiving cooperation (to choose the proper policy)
        self.coop_degrees_fixed = [0] * self.n_agents # if continuous cooperation


        # 1. fill RL_policies according to self.discrete_coop (1 or 2^(nA-1))
        # 2. Init AgentTFT
        # 3.

    def get_policies(self):
        """
        Fill with models trained via-selfplay the list of dict self.RL_policies
        for each agent:
        if self.discrete_coop: 2^(N_agents-1) per agent -> agent 0: x, 1, 2, 3, 12, 23, 13, 123
        else: only one model per agent taking as input the vector of cooperation Ct
        """


        for i in range(self.n_agents):
            if not self.discrete_coop:
                print("CONTINUOUS COOPERATION")
                # 2^(n_agents-1) models per agent according to the partition of agents in the cooperation

                # state size : addition of the cooperation vector of dim n_agents
                state_size = self.state_size + self.n_agents
                action_size = self.action_size

                for i in range(self.n_agents):
                    self.RL_policies.append([])
                    model_name = create_name_model(i, False)
                    self.RL_policies[-1][0](get_agent(state_size, action_size, model_name, model_name[:-4]))


            for j in range(self.n_agents):
                if j != j:
                    pass
                    # put the partition
                    # AgentRL(state_size, action_size, model = '')
            else:
                # only one model per agent with an observation state different (+ cooperation vector)
                pass

    def preprocess_state(self, state):
        (k,l) = IDENT_TO_COORD[self.ident]
        return preprocess_state(state, self.env_state_size[0], self.env_state_size[1], k, l)


    def act_cooperate_with(self, state, index_agents):
        nb_receivers = len(index_agents) # number of agents receiving the help
        policy = self.RL_generic_policies[nb_receivers]
        state_preprocessed = self.preprocess_state(state)
        print("state processed")
        print(state_preprocessed)
        print()
        state_preprocessed = preprocess_state_layers(state_preprocessed, self.ident, index_agents)
        print("state non flattened")
        print(state_preprocessed)
        state_preprocessed = state_preprocessed.flatten()

        action = 0
        #action = policy.act(state_preprocessed)
        self.k_steps_act_coop += 1
        return action


    def detect_cooperation(self, state, actions):
        return np.ones(self.n_agents)


    def negociation_cooperation(self):
        return np.ones(self.n_agents)

    def reaction_cooperation(self):
        # Tit-for-Tat algorithm
        # according to a cooperation detection, modify the inner graph-cooperation graph
        self.inner_coop_graph = self.inner_coop_graph

    def act(self, state):
        if self.discrete_coop:
            self.act_cooperate_with(state,self.coop_receivers)
        else:
            pass



import gym
import circular_collect
import time
import graphbasedTFT as gTFT
from agent_rlTFT import *

name_env = "circular_collect_special_4x4-v0"

env = gym.make(name_env)
env.reset()
nb_agents = len(env.agents)
auto = 0
max_steps = 10000

print(env.observation_space)
print(env.action_space)



ident = 1
agent_grTFT = []
state_size = 128
action_size = 5


agentRLTFT = Agent_RLTFT(ident, agent_grTFT, state_size, action_size, n_agents=4, env_state_size = (11,11))

s = env.reset()[0]
print(np.shape(s))

a = agentRLTFT.act_cooperate_with(s,[0,2,3])
print(a)

env.render(mode='human', highlight=False, save_fig = True)
time.sleep(2)


print('debut')

for t in range(max_steps):
    print(t)
    fig_name = './render/1/'+str(t)+'.png'
    #print(env.balls_in_room)
    env.render(mode='human', highlight=False, save_fig = True, fig_name = fig_name)
    ac = [env.action_space.sample() for _ in range(nb_agents)]
    #ac = [4]*4
    obs, r, done, _ = env.step(ac)

    o1 = obs[0]

    #print(o1)
    # print(np.shape(o))
    """
    print("agent 1")
    print(o1)

    print()
    print("agent 2")
    print(o2)
    print()

    print(r)
    """

    #print('balls_close_to_agents', env.balls_close_to_agents)
    #print('balls_on_grid', env.balls_on_grid)

    print(r)
    time.sleep(0)

print('finish')
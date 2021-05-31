import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from run_game import *


def efficiency(payoffs_agents, payoffs_nices, payoffs_defectors):
    n_agents = len(payoffs_agents)
    t_max = len(payoffs_agents[0])
    social_welfare = [sum([payoffs_agents[i][t] for i in range(n_agents)]) for t in range(t_max)]
    social_welfare_opt = [sum([payoffs_nices[i][t] for i in range(n_agents)]) for t in range(t_max)]
    social_welfare_worse = [sum([payoffs_defectors[i][t] for i in range(n_agents)]) for t in range(t_max)]
    efficiency_output = [(social_welfare[t]-social_welfare_worse[0])/(social_welfare_opt[0]-social_welfare_worse[0]) for t in range(t_max)]
    return np.round(efficiency_output[-1],3), efficiency_output


def speed(efficiency_list, delta_T=20):
    delta_T = min(len(efficiency_list), delta_T)
    print(delta_T)
    final_efficiency = efficiency_list[-1]
    print(final_efficiency)
    x = np.arange(delta_T)
    if final_efficiency == 0:
        output = 0.0
    else:
        output = metrics.auc(x, efficiency_list[:delta_T])/(delta_T*final_efficiency)
    return np.round(output, 3)


def incentive_compatibility(curve_payoffs_1nice, curve_payoffs_1egoist, payoffs_nices, payoffs_defectors):
    output = sum(curve_payoffs_1nice) - sum(curve_payoffs_1egoist)
    output /= sum(payoffs_nices) - sum(payoffs_defectors)
    return np.clip(np.round(output, 3), 0, 1)


def safety(curve_payoffs_1Agent_allDef, curve_payoffs_egoists, curve_payoffs_1Nice_allDef):
    output = sum(curve_payoffs_1Agent_allDef) - sum(curve_payoffs_egoists)
    output /= sum(curve_payoffs_egoists) - sum(curve_payoffs_1Nice_allDef)
    return np.round(output, 3)


def forgiveness(all_agents_payoffs, lateNice_vs_agents_payoffs, t_max=100, tau1 = 50, delta_T = 20):
    # adapt without integral ?
    v_late_nice_0 = lateNice_vs_agents_payoffs[tau1+1]
    v_optimal_final = all_agents_payoffs[-1]
    delta_T = min(delta_T, (t_max-tau1-1))
    cst_norm = delta_T*(v_optimal_final-v_late_nice_0)
    x = np.arange(delta_T)
    output = metrics.auc(x, lateNice_vs_agents_payoffs[tau1+1:tau1+1+delta_T])
    output -= v_late_nice_0*(delta_T-1)
    output /= cst_norm
    if abs(v_optimal_final-v_late_nice_0) < 1e-3:
        return 1.0
    else:
        return np.clip(np.round(output, 3), 0, 1)

def reset_agents(list_of_agents):
    for a in list_of_agents:
        a.reset()

def compute_metrics(env, list_agents_eval, algos_TFT_list, t_max = 100, delta_T = 20, tau1=50, metrics_fig='output_metrics.png'):

    n_agents = env.n_agents
    #coop_max_matrix = prob_matrix_to_coop_matrix(env.prob_matrix)
    parameters_TFT = algos_TFT_list[0].parameters_TFT
    coop_max_matrix = algos_TFT_list[0].max_coop_matrix
    assert len(parameters_TFT) == 4 # to deal with only TFT_Gamma
    assert len(list_agents_eval) >= n_agents

    t_coop_LN = tau1

    # CREATE n_agent Nice "TFT agents"
    nices_algos = [Agent_Nice(i, 'N', n_agents=n_agents, max_coop_matrix=coop_max_matrix) for i in range(n_agents)]

    # CREATE n_agent Egoist "TFT agents"
    egoists_algos = [Agent_Egoist(i, 'D', n_agents=n_agents, max_coop_matrix=coop_max_matrix) for i in range(n_agents)]

    # CREATE 1 LateNice "TFT agent" with index 0 : a Late Nice is a defector before t_coop_LN and cooperator after
    # we use Agent_Traitor with times [0, t_coop_LN]
    (alpha, r_in, beta, gamma) = parameters_TFT
    late_nice_algo = Agent_Traitor(0,'LateNice', max_coop_matrix=coop_max_matrix, t_traitor=[0,t_coop_LN],
                                   alpha_inertia = alpha, r_incentive = r_in, beta_adaptive = beta, gamma_proba = gamma)

    # RUNNING games with different lists of Agents to compute metrics
    N_runs = 8
    k = 0
    k_smooth = 20 # to smooth the payoffs over time
    limits_y = (0, 10)
    render = True
    name_expe = "check_metrics_1_"
    given_detection = True
    debug_rewards_IPD = True


    print()
    print("#### RUN EVAL TOURNAMENT "+str(k)+'/'+str(N_runs))
    env.reset()
    reset_agents(list_agents_eval)
    k += 1
    list_tft = [late_nice_algo] + algos_TFT_list[1:]
    for i, tft in enumerate(list_tft):
        list_agents_eval[i].agent_grTFT = tft
    payoffs_agents_LateNice = run_game(env, list_agents_eval, t_max=t_max, k_smooth=k_smooth, limits_y=limits_y,
                  render = render, name_expe = name_expe+str(k), given_detection=given_detection, debug_rewards_IPD=debug_rewards_IPD)


    print()
    print("#### RUN EVAL TOURNAMENT "+str(k)+'/'+str(N_runs))
    env.reset()
    reset_agents(list_agents_eval)
    k += 1
    list_tft = algos_TFT_list
    for i, tft in enumerate(list_tft):
        list_agents_eval[i].agent_grTFT = tft
    payoffs_agents = run_game(env, list_agents_eval, t_max=t_max, k_smooth=k_smooth, limits_y=limits_y,
                  render = render, name_expe = name_expe+str(k), given_detection=given_detection, debug_rewards_IPD=debug_rewards_IPD)



    print()
    print("#### RUN EVAL TOURNAMENT "+str(k)+'/'+str(N_runs))
    env.reset()
    reset_agents(list_agents_eval)
    k += 1
    list_tft = nices_algos
    for i, tft in enumerate(list_tft):
        list_agents_eval[i].agent_grTFT = tft
    payoffs_nices = run_game(env, list_agents_eval, t_max=t_max, k_smooth=k_smooth, limits_y=limits_y,
                  render = render, name_expe = name_expe+str(k), given_detection=given_detection, debug_rewards_IPD=debug_rewards_IPD)


    print()
    print("#### RUN EVAL TOURNAMENT " + str(k) + '/' + str(N_runs))
    env.reset()
    reset_agents(list_agents_eval)
    k += 1
    list_tft = egoists_algos
    for i, tft in enumerate(list_tft):
        list_agents_eval[i].agent_grTFT = tft
    payoffs_defectors = run_game(env, list_agents_eval, t_max=t_max, k_smooth=k_smooth, limits_y=limits_y,
                  render = render, name_expe = name_expe+str(k), given_detection=given_detection, debug_rewards_IPD=debug_rewards_IPD)


    print()
    print("#### RUN EVAL TOURNAMENT " + str(k) + '/' + str(N_runs))
    env.reset()
    reset_agents(list_agents_eval)
    k += 1
    list_tft = egoists_algos[:1] + algos_TFT_list[1:]
    for i, tft in enumerate(list_tft):
        list_agents_eval[i].agent_grTFT = tft
    payoffs_agents_1Egoist = run_game(env, list_agents_eval, t_max=t_max, k_smooth=k_smooth, limits_y=limits_y,
                  render = render, name_expe = name_expe+str(k), given_detection=given_detection, debug_rewards_IPD=debug_rewards_IPD)


    print()
    print("#### RUN EVAL TOURNAMENT " + str(k) + '/' + str(N_runs))
    env.reset()
    reset_agents(list_agents_eval)
    k += 1
    list_tft = nices_algos[:1] + algos_TFT_list[1:]
    for i, tft in enumerate(list_tft):
        list_agents_eval[i].agent_grTFT = tft
    payoffs_agents_1Nice = run_game(env, list_agents_eval, t_max=t_max, k_smooth=k_smooth, limits_y=limits_y,
                  render = render, name_expe = name_expe+str(k), given_detection=given_detection, debug_rewards_IPD=debug_rewards_IPD)


    print()
    print("#### RUN EVAL TOURNAMENT " + str(k) + '/' + str(N_runs))
    env.reset()
    reset_agents(list_agents_eval)
    k += 1
    list_tft = algos_TFT_list[:1] + egoists_algos[1:]
    for i, tft in enumerate(list_tft):
        list_agents_eval[i].agent_grTFT = tft
    payoffs_agents_1Agent_all_Def = run_game(env, list_agents_eval, t_max=t_max, k_smooth=k_smooth, limits_y=limits_y,
                  render = render, name_expe = name_expe+str(k), given_detection=given_detection, debug_rewards_IPD=debug_rewards_IPD)


    print()
    print("#### RUN EVAL TOURNAMENT " + str(k) + '/' + str(N_runs))
    env.reset()
    reset_agents(list_agents_eval)
    k += 1
    list_tft = nices_algos[:1] + egoists_algos[1:]
    for i, tft in enumerate(list_tft):
        list_agents_eval[i].agent_grTFT = tft
    payoffs_agents_1Nice_all_Def = run_game(env, list_agents_eval, t_max=t_max, k_smooth=k_smooth, limits_y=limits_y,
                  render = render, name_expe = name_expe+str(k), given_detection=given_detection, debug_rewards_IPD=debug_rewards_IPD)



    curve_payoffs_LN = payoffs_agents_LateNice[0][k_smooth:]
    curve_payoffs_agents = payoffs_agents[0][k_smooth:]
    curve_payoffs_nices = payoffs_nices[0][k_smooth:]
    curve_payoffs_1egoist = payoffs_agents_1Egoist[0][k_smooth:]
    curve_payoffs_1nice = payoffs_agents_1Nice[0][k_smooth:]
    curve_payoffs_egoists = payoffs_defectors[0][k_smooth:]
    curve_payoffs_1Agent_allDef = payoffs_agents_1Agent_all_Def[0][k_smooth:]
    curve_payoffs_1Nice_allDef = payoffs_agents_1Nice_all_Def[0][k_smooth:]


    plt.plot(curve_payoffs_LN, label='Repentant defector', color = 'purple')
    plt.plot(curve_payoffs_agents, label='Agent vs (N-1) agents', color = 'orange')
    plt.plot(curve_payoffs_nices, label='Optimal - all cooperators', color = 'green')

    plt.plot(curve_payoffs_egoists, label='Worst - all defectors', color = 'brown')
    #plt.plot(curve_payoffs_1Agent_allDef, label='Agent vs all defectors', color = 'orange')
    #plt.plot(curve_payoffs_1Nice_allDef, label='Cooperator vs all defectors', color = 'green')
    plt.plot(curve_payoffs_1nice, label='Nice vs (N-1) agents', color='pink')
    plt.plot(curve_payoffs_1egoist, label = 'Egoist vs (N-1) agents', color = 'red')

    plt.legend(loc=0)
    plt.xlabel('steps')
    plt.ylabel('payoff')
    plt.savefig(metrics_fig)
    plt.clf()

    ef, evo_efficiency = efficiency(payoffs_agents, payoffs_nices, payoffs_defectors)
    sp = speed(evo_efficiency, delta_T=delta_T)
    ic = incentive_compatibility(curve_payoffs_1nice, curve_payoffs_1egoist, curve_payoffs_nices, curve_payoffs_egoists)
    sf = safety(curve_payoffs_1Agent_allDef, curve_payoffs_egoists, curve_payoffs_1Nice_allDef)
    fg = forgiveness(payoffs_agents[0], payoffs_agents_LateNice[0], t_max=t_max, tau1=tau1, delta_T=delta_T)


    print()
    print("Efficiency = ", ef)
    print("Speed = ", sp)
    print("IC = ", ic)
    print("Safety = ", sf)
    print("Forgiveness = ", fg)

    return [ef, sp, fg, ic, sf]


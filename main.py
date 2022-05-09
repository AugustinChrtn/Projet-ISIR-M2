import numpy as np
import matplotlib.pyplot as plt
import pygame
import time 
import pandas as pd
import random

from Gridworld import State
from Useful_functions import play, find_best_duo_Q_learning, find_best_duo_Kalman_sum, find_best_trio_Q_learning, find_best_trio_Kalman_sum
from Useful_functions import plot3D, plot4D,convergence,save_pickle, open_pickle,value_iteration,policy_evaluation
from Complexworld import ComplexState
from Deterministic_world import Deterministic_State
from Uncertain_world import Uncertain_State
from Two_step_task import Two_step
from Lopesworld import Lopes_State
from Lopes_nonstat import Lopes_nostat
from Deterministic_nostat import Deterministic_no_stat
from Uncertainworld_U import Uncertain_State_U
from Uncertainworld_B import Uncertain_State_B

from Q_learning import Q_Agent
from Kalman import Kalman_agent
from Kalman_sum import Kalman_agent_sum
from Rmax import Rmax_Agent
from BEB import BEB_Agent
from KalmanMB import KalmanMB_Agent
from Q_learningMB import QMB_Agent
from BEBLP import BEBLP_Agent
from RmaxLP import RmaxLP_Agent

#Initializing parameters
environments_parameters={'Two_Step':{},'Lopes':{'transitions':np.load('Mondes/Transitions_Lopes.npy',allow_pickle=True)}}
all_environments={'Lopes':Lopes_State,'Two_Step':Two_step}
for number_world in range(1,21):
    world=np.load('Mondes/World_'+str(number_world)+'.npy')
    world_2=np.load('Mondes/World_'+str(number_world)+'_B.npy')
    transitions=np.load('Mondes/Transitions_'+str(number_world)+'.npy',allow_pickle=True)
    transitions_U=np.load('Mondes/Transitions_'+str(number_world)+'_U.npy',allow_pickle=True)
    transitions_B=np.load('Mondes/Transitions_'+str(number_world)+'_B.npy',allow_pickle=True)
    transitions_lopes=np.load('Mondes/Transitions_Lopes_non_stat'+str(number_world)+'.npy',allow_pickle=True)
    environments_parameters["D_{0}".format(number_world)] = {'world':world}
    environments_parameters["U_{0}".format(number_world)] = {'world':world,'transitions':transitions}
    environments_parameters["DB_{0}".format(number_world)] = {'world':world,'world2':world_2}
    environments_parameters["UU_{0}".format(number_world)] = {'world':world,'transitions':transitions,'transitions_U':transitions_U}
    environments_parameters["UB_{0}".format(number_world)] = {'world':world,'world2':world_2,'transitions':transitions,'transitions_B':transitions_B} 
    environments_parameters["Lopes_nostat_{0}".format(number_world)]={'transitions':np.load('Mondes/Transitions_Lopes.npy',allow_pickle=True),'transitions2':transitions_lopes}
    all_environments["D_{0}".format(number_world)]=Deterministic_State
    all_environments["U_{0}".format(number_world)]=Uncertain_State
    all_environments["DB_{0}".format(number_world)]=Deterministic_no_stat
    all_environments["UB_{0}".format(number_world)]=Uncertain_State_B
    all_environments["UU_{0}".format(number_world)]=Uncertain_State_U
    all_environments["Lopes_nostat_{0}".format(number_world)]=Lopes_nostat

seed=57
np.random.seed(seed)
random.seed(57)

agent_parameters={Q_Agent:{'alpha':0.5,'beta':0.05,'gamma':0.95,'exploration':'softmax'},
            Kalman_agent_sum:{'gamma':0.98,'variance_ob':1,'variance_tr':50,'curiosity_factor':1},
            Kalman_agent:{'gamma':0.95, 'variance_ob':1,'variance_tr':40},
            KalmanMB_Agent:{'gamma':0.95,'epsilon':0.1,'H_update':3,'entropy_factor':0.1,'epis_factor':50,'alpha':0.2,'gamma_epis':0.5,'variance_ob':0.02,'variance_tr':0.5},
            QMB_Agent:{'gamma':0.95,'epsilon':0.1,'known_states':True},
            Rmax_Agent:{'gamma':0.95, 'm':5,'Rmax':1,'known_states':True,'VI':50},
            BEB_Agent:{'gamma':0.95,'beta':2,'known_states':True,'coeff_prior':40,'informative':True},
            BEBLP_Agent:{'gamma':0.95,'beta':1.5,'step_update':10,'coeff_prior':0.001,'alpha':0.3},
            RmaxLP_Agent:{'gamma':0.95,'Rmax':1,'step_update':10,'alpha':0.2,'m':0.7,'VI':50}}


nb_iters=1
trials = 200
max_step =30
photos=[10,40,70,100,130,160,199]
screen=1
accuracy=0.05
pas_VI=50

#agents={'RA':Rmax_Agent,'BEB':BEB_Agent,'QMB':QMB_Agent,'BEBLP':BEBLP_Agent,'RALP':RmaxLP_Agent,'QA':Q_Agent,'KAS':Kalman_agent_sum,'KMB':KalmanMB_Agent}
agents={'BEBLP':BEBLP_Agent}

#environments=['Lopes_{0}'.format(num) for num in range(1,21),'Two_Step']+['D_{0}'.format(num) for num in range(1,21)]+['U_{0}'.format(num) for num in range(1,21)]

names_env=['Lopes']

rewards={(name_agent,name_environment):[] for name_agent in agents.keys() for name_environment in names_env}
steps={(name_agent,name_environment):[] for name_agent in agents.keys() for name_environment in names_env}
exploration={(name_agent,name_environment):[] for name_agent in agents.keys() for name_environment in names_env}
pol_error={(name_agent,name_environment):[] for name_agent in agents.keys() for name_environment in names_env}

for name_environment in names_env:   
    print(name_environment)
    for iteration in range(nb_iters):
        print(iteration)
        for name_agent,agent in agents.items(): 
            print(name_agent)
            environment=all_environments[name_environment](**environments_parameters[name_environment])
            
            globals()[name_agent]=agent(environment,**agent_parameters[agent]) #Defining a new agent from the dictionary agents
            
            reward,step_number,policy_value_error= play(environment,globals()[name_agent],trials=trials,max_step=max_step,photos=photos,screen=screen,accuracy=accuracy,pas_VI=pas_VI) #Playing in environment
            
            rewards[(name_agent,name_environment)].append(reward)
            steps[(name_agent,name_environment)].append(step_number)
            exploration[(name_agent,name_environment)].append(sum([len(value.keys()) for value in globals()[name_agent].counter.values()])/environment.max_exploration)
            pol_error[(name_agent,name_environment)].append(policy_value_error)
            
### Extracting results ###

#For each agent and each world

mean_rewards={(name_agent,name_environment): np.mean(rewards[(name_agent,name_environment)]) for name_agent in agents.keys() for name_environment in names_env}

mean_exploration={(name_agent,name_environment): np.mean(exploration[(name_agent,name_environment)]) for name_agent in agents.keys() for name_environment in names_env}
stats_convergence={(name_agent,name_environment):[convergence(rewards[(name_agent,name_environment)][nb_iter]) for nb_iter in range(nb_iters)]for name_agent in agents.keys() for name_environment in names_env}
avg_stats={(name_agent,name_environment): np.average(stats_convergence[(name_agent,name_environment)],axis=0)for name_agent in agents.keys() for name_environment in names_env}


trial_plateau={(name_agent,name_environment):np.array(stats_convergence[(name_agent,name_environment)])[:,0] for name_agent in agents.keys() for name_environment in names_env}
step_plateau={(name_agent,name_environment):[steps[(name_agent,name_environment)][nb_iter][int(trial_plateau[(name_agent,name_environment)][nb_iter])] for nb_iter in range(nb_iters)]for name_agent in agents.keys() for name_environment in names_env}
avg_step_plateau={(name_agent,name_environment): np.average(step_plateau[(name_agent,name_environment)]) for name_agent in agents.keys() for name_environment in names_env}

min_length=np.min([len(pol_error[name_agent,name_environment][i]) for i in range(nb_iters) for name_agent in agents.keys() for name_environment in names_env])
mean_pol_error={(name_agent,name_environment):np.average([pol_error[name_agent,name_environment][i][:min_length] for i in range(nb_iters)],axis=0) for name_environment in names_env for name_agent in agents.keys()}
#For each agent
mean_reward_agent={name_agent: np.mean([mean_rewards[(name_agent,name_environment)]for name_environment in names_env]) for name_agent in agents.keys() }

mean_exploration_agent={name_agent: np.mean([mean_exploration[(name_agent,name_environment)]for name_environment in names_env]) for name_agent in agents.keys() }

stats_agent={name_agent:np.average([avg_stats[(name_agent,name_environment)] for name_environment in names_env],axis=0) for name_agent in agents.keys()}
step_plateau_agent={name_agent: np.average([avg_step_plateau[(name_agent,name_environment)] for name_environment in names_env],axis=0) for name_agent in agents.keys()}
mean_pol_error_agent={name_agent: np.average([np.average([pol_error[name_agent,name_environment][i][:min_length] for i in range(nb_iters)],axis=0) for name_environment in names_env],axis=0) for name_agent in agents.keys()}

print("")
for name_agent in agents.keys():
    print(name_agent+' : '+ 'avg_reward= '+str(round(mean_reward_agent[name_agent],2))+", trial_conv= "+str(stats_agent[name_agent][0])+
          ', step_conv= '+str(round(step_plateau_agent[name_agent]))+
          ', mean= '+str(round(stats_agent[name_agent][1]))+', var= '+str(round(stats_agent[name_agent][2]))+', explo= '+str(round(mean_exploration_agent[name_agent],3)))
    print("")


###Basic visualisation ###

rewards_agent_environment={(name_agent,name_environment): np.average(np.array(rewards[name_agent,name_environment]),axis=0) for name_environment in names_env for name_agent in agents.keys()}
rewards_agent={name_agent: np.average(np.average(np.array([rewards[name_agent,name_environment] for name_environment in names_env]),axis=1),axis=0) for name_agent in agents.keys()}



for name_agent in agents.keys():
    avg_reward=rewards_agent[name_agent]
    plt.figure()
    plt.plot(avg_reward, color='black')
    plt.xlabel("Trial")
    plt.ylabel("Reward")
    plt.title(name_agent)
    plt.show()

    


plt.figure()
for name_agent in agents.keys():
    plt.plot([pas_VI*i for i in range(min_length)],mean_pol_error_agent[name_agent],label=name_agent)
plt.xlabel("Steps")
plt.ylabel("Policy value error")
plt.legend()
plt.show()


pygame.quit()


### Save results ###

results={'seed':seed,'nb_iters':nb_iters,'trials':trials,'max_step':max_step,'agent_parameters':agent_parameters,'agents':agents,'environments':names_env,'rewards':rewards,'step_number':step_number,'pol_error':pol_error}

temps=str(round(time.time()))
save_pickle(results,'Results/'+temps+'.pickle')
#test=open_pickle('Results/'+temps+'.pickle')




### Parameter fitting ###

"""
table_Q2=find_best_duo_Q_learning(number_environment=1)
np.save('Data/Q_duo'+str(time.time())+'.npy',table_Q2)
plot3D(table_Q2)"""

"""
table_Q=find_best_trio_Q_learning(number_environment=1)
np.save('Data/Q_trio'+str(time.time())+'.npy',table_Q)
plot4D(table_Q)
"""
"""
table_KAS=find_best_duo_Kalman_sum(number_environment=1,uncertain=False)
np.save('Data/KAS_trio'+str(time.time())+'.npy',table_KAS)
plot3D(table_KAS,x_name='variance_tr',y_name='curiosity_factor')
"""
"""
table_KAS2=find_best_trio_Kalman_sum(number_environment=1,uncertain=False)
np.save('Data/KAS2_trio'+str(time.time())+'.npy',table_KAS2)
plot4D(table_KAS2,x_name='variance_tr',y_name='curiosity_factor',z_name='gamma')
"""

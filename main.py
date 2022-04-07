import numpy as np
import matplotlib.pyplot as plt
import pygame
import time 

from Gridworld import State
from Useful_functions import play, find_best_duo_Q_learning, find_best_duo_Kalman_sum, find_best_trio_Q_learning, find_best_trio_Kalman_sum
from Useful_functions import plot3D, plot4D,convergence

from Complexworld import ComplexState
from Deterministic_world import Deterministic_State
from Uncertain_world import Uncertain_State
from Two_step_task import Two_step
from Lopesworld import Lopes_State

from Q_learning import Q_Agent
from Kalman import Kalman_agent
from Kalman_sum import Kalman_agent_sum
from Rmax import Rmax_Agent
from BEB import BEB_Agent
from KalmanMB import KalmanMB_Agent
from Q_learningMB import QMB_Agent

#Initializing parameters
environments_parameters={'Lopes':{'transitions':np.load('Mondes/Transitions_Lopes.npy',allow_pickle=True)},'Two_Step':{}}
all_environments={'Lopes':Lopes_State,'Two_Step':Two_step}
for number_world in range(1,21):
    world=np.load('Mondes/World_'+str(number_world)+'.npy')
    transitions=np.load('Mondes/Transitions_'+str(number_world)+'.npy',allow_pickle=True)
    environments_parameters["D_{0}".format(number_world)] = {'world':world}
    environments_parameters["U_{0}".format(number_world)] = {'world':world,'transitions':transitions}
    all_environments["D_{0}".format(number_world)]=Deterministic_State
    all_environments["U_{0}".format(number_world)]=Uncertain_State

seed=53

agent_parameters={Q_Agent:{'alpha':0.5,'beta':0.05,'gamma':0.95,'exploration':'softmax'},
            Kalman_agent_sum:{'gamma':0.98,'variance_ob':1,'variance_tr':50,'curiosity_factor':1},
            Rmax_Agent:{'gamma':0.9, 'max_visits_per_state':8, 'epsilon' :0.1,'Rmax':200},
            BEB_Agent:{'gamma':0.9,'beta':500},
            KalmanMB_Agent:{'gamma':0.95,'epsilon':0.1,'H_update':3,'entropy_factor':0.1,'epis_factor':50,'alpha':0.2,'gamma_epis':0.5,'variance_ob':0.02,'variance_tr':0.5},
            QMB_Agent:{'gamma':0.95,'epsilon':0.1,'H_update':5,'entropy_factor':0,'gamma_epis':0.5,'epis_factor':50},
            Kalman_agent:{'gamma':1, 'variance_ob':1,'variance_tr':40}}
nb_iters=1
trials = 1000
max_step = 500

#agents={'QA':Q_Agent,'KAS':Kalman_agent_sum,'RA':Rmax_Agent,'BEB':BEB_Agent,'KMB':KalmanMB_Agent,'QMB':QMB_Agent}
agents={'QA':Q_Agent}

#environments=['Lopes',Two_Step']+['D_{0}'.format(num) for num in range(1,21)]+['U_{0}'.format(num) for num in range(1,21)]

names_env=['U_{0}'.format(num) for num in range(1,2)]

rewards={(name_agent,name_environment):[] for name_agent in agents.keys() for name_environment in names_env}
steps={(name_agent,name_environment):[] for name_agent in agents.keys() for name_environment in names_env}

for name_environment in names_env:
    
    print(name_environment)
    environment=all_environments[name_environment](**environments_parameters[name_environment])
        
    for iteration in range(nb_iters):
        for name_agent,agent in agents.items(): 
            
            globals()[name_agent]=agent(environment,**agent_parameters[agent]) #Defining a new agent from the dictionary agents
            
            reward,step_number= play(environment,globals()[name_agent],trials=trials,max_step=max_step) #Playing in environment
            
            rewards[(name_agent,name_environment)].append(reward)
            steps[(name_agent,name_environment)].append(step_number)

            
### Extracting results ###

#For each agent and each world

mean_rewards={(name_agent,name_environment): np.mean(rewards[(name_agent,name_environment)]) for name_agent in agents.keys() for name_environment in names_env}

stats_convergence={(name_agent,name_environment):[convergence(rewards[(name_agent,name_environment)][nb_iter]) for nb_iter in range(nb_iters)]for name_agent in agents.keys() for name_environment in names_env}
avg_stats={(name_agent,name_environment): np.average(stats_convergence[(name_agent,name_environment)],axis=0)for name_agent in agents.keys() for name_environment in names_env}


trial_plateau={(name_agent,name_environment):np.array(stats_convergence[(name_agent,name_environment)])[:,0] for name_agent in agents.keys() for name_environment in names_env}
step_plateau={(name_agent,name_environment):[steps[(name_agent,name_environment)][nb_iter][int(trial_plateau[(name_agent,name_environment)][nb_iter])] for nb_iter in range(nb_iters)]for name_agent in agents.keys() for name_environment in names_env}
avg_step_plateau={(name_agent,name_environment): np.average(step_plateau[(name_agent,name_environment)]) for name_agent in agents.keys() for name_environment in names_env}



#For each agent
mean_reward_agent={name_agent: np.mean([mean_rewards[(name_agent,name_environment)]for name_environment in names_env]) for name_agent in agents.keys() }
stats_agent={name_agent:np.average([avg_stats[(name_agent,name_environment)] for name_environment in names_env],axis=0) for name_agent in agents.keys()}
step_plateau_agent={name_agent: np.average([avg_step_plateau[(name_agent,name_environment)] for name_environment in names_env],axis=0) for name_agent in agents.keys()}

print("")
for name_agent in agents.keys():
    print(name_agent+' : '+ 'avg_reward= '+str(round(mean_reward_agent[name_agent],2))+", trial_conv= "+str(stats_agent[name_agent][0])+
          ', step_conv= '+str(round(step_plateau_agent[name_agent]))+
          ', mean= '+str(round(stats_agent[name_agent][1]))+', var= '+str(round(stats_agent[name_agent][2])))
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

pygame.quit()




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

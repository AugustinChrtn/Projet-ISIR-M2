
import numpy as np
import matplotlib.pyplot as plt
import pygame
import seaborn as sns
import time 

from Gridworld import State
from Useful_functions import play, find_best_duo_Q_learning, find_best_duo_Kalman_sum, find_best_trio_Q_learning, find_best_trio_Kalman_sum
from Useful_functions import plot3D, plot4D,plateau
from Complexworld import ComplexState
from Deterministic_world import Deterministic_State
from Uncertain_world import Uncertain_State
from Two_step_task import Two_step

from Q_learning import Q_Agent
from Kalman import Kalman_agent
from Kalman_sum import Kalman_agent_sum
from Rmax import Rmax_Agent
from BEB import BEB_Agent
from KalmanMB import KalmanMB_Agent

for number_world in range(2,3):

    world=np.load('Mondes/World_'+str(number_world)+'.npy')
    transitions=np.load('Mondes/Transitions_'+str(number_world)+'.npy',allow_pickle=True)
    
    environment=Uncertain_State(world, transitions)
    trials, max_step=500,500
    nb_agents=5
    nb_iters=1
    
    rewards=[[]for i in range(nb_agents)]
    step_plateau=[[] for i in range(nb_agents)]
    for i in range(nb_iters):
        
            print(i)
            
            QA=Q_Agent(environment,alpha=0.5,beta=0.05,gamma=0.95,exploration='softmax')
            KAS=Kalman_agent_sum(environment,gamma=0.98,variance_ob=1,variance_tr=50,curiosity_factor=1)
            RA=Rmax_Agent(environment,gamma=0.9, max_visits_per_state=10, epsilon =0.1,Rmax=200)
            BEB=BEB_Agent(environment,gamma=0.9,beta=500)
            KMB=KalmanMB_Agent(environment,gamma=0.95,epsilon = 0.1,H_update=3,entropy_factor=0.1,epis_factor=50,alpha=0.2,gamma_epis=0.5,variance_ob=0.02,variance_tr=1)
            
            reward_QA,step_number_QA, counter_QA,value_QA = play(environment, QA,trials=trials,max_step=max_step)
            reward_KAS,step_number_KAS, counter_KAS, table_mean_KAS,table_variance_KAS = play(environment,KAS,trials=trials,max_step=max_step)
            reward_RA, step_number_RA, counter_RA, value_RA, transitions_model_RA, reward_model_RA=play(environment,RA,trials=trials,max_step=max_step)
            reward_BEB, step_number_BEB, counter_BEB, value_BEB, transitions_model_BEB, reward_model_BEB=play(environment,BEB,trials=trials,max_step=max_step)           
            reward_KMB,step_number_KMB,counter_KMB,value_KMB,transitions_model_KMB,reward_model_KMB=play(environment,KMB,trials=trials,max_step=max_step)
            
            rewards[0].append(reward_QA)
            rewards[1].append(reward_KAS)
            rewards[2].append(reward_RA)
            rewards[3].append(reward_BEB)            
            rewards[4].append(reward_KMB)
            
            step_plateau[0].append(step_number_QA)
            step_plateau[1].append(step_number_KAS)
            step_plateau[2].append(step_number_RA)
            step_plateau[3].append(step_number_BEB)           
            step_plateau[4].append(step_number_KMB)
            
mean_rewards=[np.mean(reward) for reward in rewards]
plateaux=[[plateau(rewards[nb_agent][nb_iter]) for nb_iter in range(nb_iters)]for nb_agent in range(nb_agents)]
avg_plateaux=np.average(plateaux,axis=1)

peaks=np.array(plateaux)[:,:,0]
number_steps=[[step_plateau[nb_agent][nb_iter][int(peaks[nb_agent][nb_iter])] for nb_iter in range(nb_iters)]for nb_agent in range(nb_agents)]
avg_steps=np.average(number_steps,axis=1)
print("avg_reward_QA = "+str(round(mean_rewards[0]))+", plateau_QA= "+str(avg_plateaux[0][0])+', mean_QA= '+str(round(avg_plateaux[0][1]))+', var_QA= '+str(round(avg_plateaux[0][2]))+', step_QA= '+str(round(avg_steps[0])))
print("avg_reward_KAS = "+str(round(mean_rewards[1]))+", plateau_KAS= "+str(avg_plateaux[1][0])+', mean_KAS= '+str(round(avg_plateaux[1][1]))+', var_KAS= '+str(round(avg_plateaux[1][2]))+', step_KAS= '+str(round(avg_steps[1])))
print("avg_reward_RA = "+str(round(mean_rewards[2]))+", plateau_RA= "+str(avg_plateaux[2][0])+', mean_RA= '+str(round(avg_plateaux[2][1]))+', var_RA= '+str(round(avg_plateaux[2][2]))+', step_RA= '+str(round(avg_steps[2])))
print("avg_reward_BEB = "+str(round(mean_rewards[3]))+", plateau_BEB= "+str(avg_plateaux[3][0])+', mean_BEB= '+str(round(avg_plateaux[3][1]))+', var_BEB= '+str(round(avg_plateaux[3][2]))+', step_BEB= '+str(round(avg_steps[3])))
print("avg_reward_KMB = "+str(round(mean_rewards[4]))+", plateau_KMB= "+str(avg_plateaux[4][0])+', mean_KMB= '+str(round(avg_plateaux[4][1]))+', var_KMB= '+str(round(avg_plateaux[4][2]))+', step_KMB= '+str(round(avg_steps[4])))
print(" ")
  
avg_reward_QA=np.average(np.array(rewards[0]),axis=0)
avg_reward_KAS=np.average(np.array(rewards[1]),axis=0)
avg_reward_RA=np.average(np.array(rewards[2]),axis=0)
avg_reward_BEB=np.average(np.array(rewards[3]),axis=0)
avg_reward_KMB=np.average(np.array(rewards[4]),axis=0)

#One trial
"""
plt.figure()
plt.plot(reward_QA, color='black')
plt.xlabel("Essai Q-learning softmax")
plt.ylabel("Récompense")
plt.show()

plt.figure()
plt.plot(reward_KAS, color='black')
plt.xlabel("Essai KAS")
plt.ylabel("Récompense")
plt.show()

plt.figure()
plt.plot(reward_RA,color='black')
plt.xlabel("Essai RA")
plt.ylabel("Récompense")
plt.show()

plt.figure()
plt.plot(reward_BEB,color='black')
plt.xlabel("Essai BEB")
plt.ylabel("Récompense")
plt.show()
"""

#Average over trials

plt.figure()
plt.plot(avg_reward_QA, color='black')
plt.xlabel("Essai Q-learning softmax")
plt.ylabel("Récompense")
plt.show()

plt.figure()
plt.plot(avg_reward_KAS, color='black')
plt.xlabel("Essai KAS")
plt.ylabel("Récompense")
plt.show()

plt.figure()
plt.plot(avg_reward_RA, color='black')
plt.xlabel("Essai RA")
plt.ylabel("Récompense")
plt.show()

plt.figure()
plt.plot(avg_reward_BEB, color='black')
plt.xlabel("Essai BEB")
plt.ylabel("Récompense")
plt.show()

plt.figure()
plt.plot(avg_reward_KMB, color='black')
plt.xlabel("Essai KMB")
plt.ylabel("Récompense")
plt.show()

pygame.quit()


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


import numpy as np
import matplotlib.pyplot as plt
import pygame
import seaborn as sns
import time 

from Gridworld import State
from Useful_functions import play, find_best_duo_Q_learning, find_best_duo_Kalman_sum, find_best_trio_Q_learning, find_best_trio_Kalman_sum, plot3D, plot4D
from Complexworld import ComplexState
from Deterministic_world import Deterministic_State
from Uncertain_world import Uncertain_State

from Q_learning import Q_Agent
from Kalman import Kalman_agent
from Kalman_sum import Kalman_agent_sum
from Rmax import Rmax_Agent

world=np.load('Mondes/World_10.npy')
transitions=np.load('Mondes/Transitions_10.npy',allow_pickle=True)

environment=Uncertain_State(world,transitions)
rewards=[[]for i in range(3)]

for i in range(1):
        print(i)
        QA=Q_Agent(environment,alpha=0.95,beta=1,gamma=0.985)
        KAS=Kalman_agent_sum(environment,gamma=0.95,variance_ob=1,variance_tr=50,curiosity_factor=2)
        RA=Rmax_Agent(environment,gamma=0.95, max_visits_per_state=10, epsilon = 0.1)
        
        reward_QA,counter_QA,value_QA = play(environment, QA)
        reward_KAS,counter_KAS, table_mean_KAS,table_variance_KAS = play(environment,KAS)
        reward_RA, counter_RA, value_RA, transitions_model_RA, reward_model_RA=play(environment,RA)
      
        rewards[0].append(reward_QA)
        rewards[1].append(reward_KAS)
        rewards[2].append(reward_RA)
 
mean_rewards=[np.mean(reward) for reward in rewards]
print("avg_reward_total_QA = "+str(mean_rewards[0]))
print("avg_reward_total_KA = "+str(mean_rewards[1]))
print("avg_reward_total_RA = "+str(mean_rewards[2]))
print(" ")
  
avg_reward_QA=np.average(np.array(rewards[0]),axis=0)
avg_reward_KAS=np.average(np.array(rewards[1]),axis=0)
avg_reward_RA=np.average(np.array(rewards[2]),axis=0)

#One trial

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
plt.plot(reward_RA, color='black')
plt.xlabel("Essai RA")
plt.ylabel("Récompense")
plt.show()

#Average over trials
"""
plt.figure()
plt.plot(avg_reward_QA, color='black')
plt.xlabel("Essai Q-learning softmax")
plt.ylabel("Récompense")
plt.show()

plt.figure()
plt.plot(avg_reward_KAS, color='black')
plt.xlabel("Essai KAS")
plt.ylabel("Récompense")
plt.show()"""
 
pygame.quit()

table_Q2=find_best_duo_Q_learning(number_environment=1)
np.save('Data/Q_duo'+str(time.time())+'.npy',table_Q2)
plot3D(table_Q2)

"""
table_Q=find_best_trio_Q_learning(number_environment=1)
np.save('Data/Q_trio'+str(time.time())+'.npy',table_Q)
plot4D(table_Q)"""


"""
table_KAS=find_best_duo_Kalman_sum(number_environment=1,uncertain=False)
np.save('Data/KAS_trio'+str(time.time())+'.npy',table_KAS)
plot3D(table_KAS,x_name='variance_tr',y_name='curiosity_factor')

table_KAS2=find_best_trio_Kalman_sum(number_environment=1,uncertain=False)
np.save('Data/KAS_trio'+str(time.time())+'.npy',table_KAS)
plot4D(table_KAS,x_name='curiosity_factor',y_name='variance_tr',z_name='gamma')"""

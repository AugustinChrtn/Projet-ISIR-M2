
import numpy as np
import matplotlib.pyplot as plt
import pygame
import seaborn as sns
import time 

from Gridworld import State
from Useful_functions import play, find_best_duo_Q_learning, find_best_duo_Kalman_sum, find_best_trio_Q_learning, plot3D, plot4D
from Complexworld import ComplexState
from Deterministic_world import Deterministic_State
from Uncertain_world import Uncertain_State

from Q_learning import Q_Agent
from Kalman import Kalman_agent
from Kalman_sum import Kalman_agent_sum

world=np.load('Mondes/World_10.npy')
transitions=np.load('Mondes/Transitions_10.npy',allow_pickle=True)

environment=Uncertain_State(world,transitions)
rewards=[[]for i in range(2)]
for i in range(1):
        print(i)
        QA=Q_Agent(environment,alpha=0.95,beta=1,gamma=0.985)
        KAS=Kalman_agent_sum(environment,gamma=0.98,variance_ob=1,variance_tr=50,curiosity_factor=1)
        
        reward_QA,counter_QA,value_QA = play(environment, QA)
        reward_KAS,counter_KAS, table_mean_KAS,table_variance_KAS = play(environment,KAS)
      
        rewards[0].append(reward_QA)
        rewards[1].append(reward_KAS)
    
mean_rewards=[np.mean(reward) for reward in rewards]
print("avg_reward_total_QA = "+str(mean_rewards[0]))
print("avg_reward_total_KA = "+str(mean_rewards[1]))
print(" ")
  
avg_reward_QA=np.average(np.array(rewards[0]),axis=0)
avg_reward_KAS=np.average(np.array(rewards[1]),axis=0)
  

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
plt.show()"""

pygame.quit()

table=find_best_trio_Q_learning(number_environment=1)
np.save('Data/Q_trio'+str(time.time())+'.npy',table)
plot4D(table)

"""
table_2=find_best_duo_Q_learning(number_environment=1)
np.save('Data/Q_duo'+str(time.time())+'.npy',table_2)
plot3D(table_2)"""

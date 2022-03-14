
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
from Two_step_task import Two_step

from Q_learning import Q_Agent
from Kalman import Kalman_agent
from Kalman_sum import Kalman_agent_sum
from Rmax import Rmax_Agent
from BEB import BEB_Agent

world=np.load('Mondes/World_10.npy')
transitions=np.load('Mondes/Transitions_10.npy',allow_pickle=True)

environment=Uncertain_State(world, transitions)
trials, max_step=1000,500

rewards=[[]for i in range(4)]

for i in range(1):
        print(i)
        QA=Q_Agent(environment,alpha=0.2,beta=5,gamma=0.95,exploration='softmax')
        KAS=Kalman_agent_sum(environment,gamma=0.98,variance_ob=0.01,variance_tr=0.01,curiosity_factor=0)
        RA=Rmax_Agent(environment,gamma=0.95, max_visits_per_state=3, epsilon = 0.1,Rmax=1)
        BEB=BEB_Agent(environment,gamma=0.95,beta=1)
        
        reward_QA,counter_QA,value_QA = play(environment, QA,trials=trials,max_step=max_step)
        reward_KAS,counter_KAS, table_mean_KAS,table_variance_KAS = play(environment,KAS,trials=trials,max_step=max_step)
        reward_RA, counter_RA, value_RA, transitions_model_RA, reward_model_RA=play(environment,RA,trials=trials,max_step=max_step)
        reward_BEB, counter_BEB, value_BEB, transitions_model_BEB, reward_model_BEB=play(environment,BEB,trials=trials,max_step=max_step)
      
        rewards[0].append(reward_QA)
        rewards[1].append(reward_KAS)
        rewards[2].append(reward_RA)
        rewards[3].append(reward_BEB)
 
mean_rewards=[np.mean(reward) for reward in rewards]
print("avg_reward_total_QA = "+str(mean_rewards[0]))
print("avg_reward_total_KA = "+str(mean_rewards[1]))
print("avg_reward_total_RA = "+str(mean_rewards[2]))
print("avg_reward_total_BEB = "+str(mean_rewards[3]))
print(" ")
  
avg_reward_QA=np.average(np.array(rewards[0]),axis=0)
avg_reward_KAS=np.average(np.array(rewards[1]),axis=0)
avg_reward_RA=np.average(np.array(rewards[2]),axis=0)
avg_reward_BEB=np.average(np.array(rewards[3]),axis=0)

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
plt.plot(reward_RA,color='black')
plt.xlabel("Essai RA")
plt.ylabel("Récompense")
plt.show()

plt.figure()
plt.plot(reward_BEB,color='black')
plt.xlabel("Essai BEB")
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
"""
table_Q2=find_best_duo_Q_learning(number_environment=1)
np.save('Data/Q_duo'+str(time.time())+'.npy',table_Q2)
plot3D(table_Q2)
"""
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

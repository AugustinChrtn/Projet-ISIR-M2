import numpy as np
import copy
import pygame
import seaborn as sns
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Gridworld import State
from Complexworld import ComplexState
from Deterministic_world import Deterministic_State
from Uncertain_world import Uncertain_State

from Q_learning import Q_Agent
from Kalman import Kalman_agent
from Kalman_sum import Kalman_agent_sum
from Rmax import Rmax_Agent


from Representation import Graphique



#MAIN

def play(environment, agent, trials=200, max_step=500, screen=0,photos=[50,100,150,199]):
    reward_per_episode = []
    for trial in range(trials):
        if screen :
            if trial in photos:
                if type(agent).__name__=='Q_Agent': 
                    value=copy.deepcopy(agent.Q)
                if type(agent).__name__ in ['Kalman_agent','Kalman_agent_sum']: 
                    value=copy.deepcopy(agent.K_mean)
                    if type(agent).__name__ =='Kalman_agent_sum':
                        curiosity=copy.deepcopy(agent.K_var)
                        img2=picture_world(environment,curiosity)
                        pygame.image.save(img2.screen,"Images/curiosity_"+type(agent).__name__+str(trial)+".png")
                if type(agent).__name__!='Rmax_Agent':
                    img=picture_world(environment,value)
                    pygame.image.save(img.screen,"Images/"+type(agent).__name__+"_"+str(trial)+".png")
        
        cumulative_reward, step, game_over= 0,0,False
        while step < max_step and game_over != True:
            old_state = environment.current_location
            action = agent.choose_action() 
            reward , terminal = environment.make_step(action)
            new_state = environment.current_location            
            agent.learn(old_state, reward, new_state, action)                
            cumulative_reward += reward
            step += 1            
            if terminal == True :
                environment.current_location=environment.first_location
                game_over = True 
        if step == max_step : environment.current_location=environment.first_location                    
        reward_per_episode.append(cumulative_reward)
    if type(agent).__name__=='Q_Agent': return reward_per_episode, agent.counter, agent.Q
    if type(agent).__name__=='Kalman_agent': return reward_per_episode, agent.counter, agent.K_mean, agent.K_var
    if type(agent).__name__=='Kalman_agent_sum': return reward_per_episode,agent.counter,agent.K_mean,agent.K_var
    if type(agent).__name__=='Rmax_Agent': return reward_per_episode,agent.counter,agent.qSA,agent.tSAS,agent.R
    if type(agent).__name__=='BEB_Agent': return reward_per_episode,agent.counter,agent.qSA,agent.tSAS,agent.R

###### OPTIMISATION ######

def find_best_duo_Q_learning(number_environment,alpha=0.6,uncertain=False):
    
    beta_range=[np.around(2*j*1e-4,decimals=5) for j in range(10,101,10)]
    gamma_range=[(0.95+i/100) for i in range(5)]
    
    world=np.load('Mondes/World_10.npy')
    transitions=np.load('Mondes/Transitions_10.npy',allow_pickle=True)
    
    results=[]
    for index_beta,beta in enumerate(beta_range):
        print(index_beta) 
        for gamma in gamma_range:
                if uncertain : environment=Uncertain_State(world,transitions)
                else : environment=Deterministic_State(world)
                QA=Q_Agent(environment,alpha,beta,gamma)
                reward_per_episode, counter_QA, Q_values= play(environment,QA,trials=200)
                results.append([beta,gamma,np.mean(reward_per_episode)])
    return np.array(results)



def find_best_duo_Kalman_sum(number_environment,uncertain=False):
    
    variance_tr_range=[np.around(j*10,decimals=4) for j in range(1,21,2)]
    curiosity_factor_range=[np.around(j*1e-1,decimals=5) for j in range(1,21,2)] 
    
    world=np.load('Mondes/World_'+str(number_environment)+'.npy')
    transitions=np.load('Mondes/Transitions_'+str(number_environment)+'.npy',allow_pickle=True)
    
    results=[]
    for index_variance_tr,variance_tr in enumerate(variance_tr_range):
        print(index_variance_tr) 
        for curiosity_factor in curiosity_factor_range:
                if uncertain : environment=Uncertain_State(world,transitions)
                else : environment = Deterministic_State(world)
                KAS=Kalman_agent_sum(environment,gamma=0.99,variance_ob=1,variance_tr=variance_tr,curiosity_factor=curiosity_factor)
                reward_per_episode,counter_KAS, table_mean_KAS,table_variance_KAS = play(environment,KAS,trials=200)
                results.append([variance_tr,curiosity_factor,np.mean(reward_per_episode)])
    return np.array(results)




def find_best_trio_Q_learning(number_environment,uncertain=False):
    alpha_range=[np.around(j*1e-2,decimals=4) for j in range(40,81,4)]
    beta_range=[np.around(2*j*1e-4,decimals=5) for j in range(10,101,10)]
    gamma_range=[(0.9+i/100) for i in range(3)]    
    
    world=np.load('Mondes/World_'+str(number_environment)+'.npy')
    transitions=np.load('Mondes/Transitions_'+str(number_environment)+'.npy',allow_pickle=True)
    
    results=[]
    for index_alpha,alpha in enumerate(alpha_range):
        print(index_alpha) 
        for beta in beta_range:
            for gamma in gamma_range:
                if uncertain : environment=Uncertain_State(world,transitions)
                else : environment = Deterministic_State(world)
                QA= Q_Agent(environment,alpha,beta,gamma)
                reward_per_episode, counter_QA, Q_values= play(environment,QA,trials=200)
                results.append([alpha,beta,gamma,np.mean(reward_per_episode)])
    return np.array(results)
           

def find_best_trio_Kalman_sum(number_environment,uncertain=False):
    variance_tr_range=[np.around(j*10,decimals=4) for j in range(1,21,2)]
    curiosity_factor_range=[np.around(j*1e-1,decimals=5) for j in range(1,21,2)] 
    gamma_range=[np.around(0.9+i/50,decimals=4) for i in range(5)]   
    
    world=np.load('Mondes/World_'+str(number_environment)+'.npy')
    transitions=np.load('Mondes/Transitions_'+str(number_environment)+'.npy',allow_pickle=True)  
    
    results=[]
    
    for index_variance_tr,variance_tr in enumerate(variance_tr_range):
        print(index_variance_tr) 
        for curiosity_factor in curiosity_factor_range:
            for gamma in gamma_range : 
                if uncertain : environment=Uncertain_State(world,transitions)
                else : environment = Deterministic_State(world)
                KAS=Kalman_agent_sum(environment,gamma=gamma,variance_ob=1,variance_tr=variance_tr,curiosity_factor=curiosity_factor)
                reward_per_episode,counter_KAS, table_mean_KAS,table_variance_KAS = play(environment,KAS,trials=200)
                results.append([variance_tr,curiosity_factor,gamma,np.mean(reward_per_episode)])
    return np.array(results)


######## VISUALISATION ########

def convert_from_default(dic):
    from collections import defaultdict
    if isinstance(dic,defaultdict):
        return dict((key,convert_from_default(value)) for key,value in dic.items())
    else : return dic

def normalized_table(table,environment):
     precision=np.array([[0 for i in range(environment.height)] for j in range(environment.width)])
     for i in range(environment.height):
         for j in range(environment.width):
             max_table=max(table[i,j])
             precision[i][j]=max_table
     mini=np.min(precision)
     if mini < 0 : precision-=mini
     normalization_factor=np.max(precision)
     if normalization_factor !=0 : precision=precision/normalization_factor     
     return precision

 
def picture_world(environment,table):       
    precision=normalized_table(table,environment)
    init_loc=[environment.first_location[0],environment.first_location[1]]
    screen_size = environment.height*50
    cell_width = 44.8
    cell_height = 44.8
    cell_margin = 5
    gridworld = Graphique(screen_size,cell_width, cell_height, cell_margin,environment.grid,environment.final_states,init_loc,precision)
    return gridworld


def plot3D(table_3D,x_name='beta',y_name='gamma'):
    dataframe=pd.DataFrame({x_name:table_3D[:,0], y_name:table_3D[:,1], 'values':table_3D[:,2]})
    data_pivoted = dataframe.pivot(x_name, y_name, "values")
    sns.heatmap(data_pivoted,cmap='Blues')
    

def plot4D(table_4D,x_name='alpha',y_name='beta',z_name='gamma',save=False): #Changer en %matplotlib auto
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(table_4D[:,0], table_4D[:,1], table_4D[:,2], c=table_4D[:,3],s=50,cmap='Blues')
    fig.colorbar(img,location='left')
    ax.set_xlabel(x_name), ax.set_ylabel(y_name), ax.set_zlabel(z_name)
    fig.show()    
    
    values=table_4D[:,3]
    color=[]
    for value in values : 
        c=0
        if value > 100 : c=100
        elif value>50: c = 50
        elif value >15 : c=15
        color.append(c)
    table2=table_4D[table_4D[:,3]>15]
    color=np.array(color)
    color=color[color>0]
    
    fig2 = plt.figure()    
    ax = fig2.add_subplot(111, projection='3d')
    img = ax.scatter(table2[:,0], table2[:,1], table2[:,2], c=color,s=50,cmap=plt.get_cmap('Set3', 3))
    clb=fig2.colorbar(img,location='left',ticks=np.array([30,57.8,85]))
    clb.set_ticklabels(['> 15','> 50','> 100'])
    ax.set_xlabel(x_name), ax.set_ylabel(y_name), ax.set_zlabel(z_name)
    #save_interactive(fig2,str(time.time()))
    fig2.show()
    print(table_4D[table_4D[:,3]>50])
    
    
def save_interactive(fig,name):
    pickle.dump(fig, open('Interactive/'+name+'.fig.pickle', 'wb'))

def plot_interactive(name_fig):
    figx=pickle.load(open('Interactive/'+name_fig,'rb'))
    figx.show()
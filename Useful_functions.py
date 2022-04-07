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

def play(environment, agent, trials=200, max_step=500, screen=1,photos=[10,20,50,100,199,300,499]):
    reward_per_episode = []
    step_number=[]
    for trial in range(trials):
        
        if screen : take_picture(agent,trial,environment,photos) #Visualisation
        
        cumulative_reward, step, game_over= 0,0,False
        while not game_over :
            old_state = environment.current_location
            action = agent.choose_action() 
            reward , terminal = environment.make_step(action) #reward and if state is terminal
            new_state = environment.current_location            
            agent.learn(old_state, reward, new_state, action)                
            cumulative_reward += reward
            step += 1            
            if terminal == True or step==max_step:
                game_over = True
                environment.current_location=environment.first_location
                step_number.append(agent.step_counter)
        reward_per_episode.append(cumulative_reward)
    return reward_per_episode,step_number



### PICTURES ###

def take_picture(agent,trial,environment,photos):
            if trial in photos:
                    value=copy.deepcopy(agent.Q)
                    img=picture_world(environment,value)
                    pygame.image.save(img.screen,"Images/"+type(agent).__name__+"_"+str(trial)+".png")
                    if type(agent).__name__ =='Kalman_agent_sum':
                        curiosity=copy.deepcopy(agent.K_var)
                        img2=picture_world(environment,curiosity)
                        pygame.image.save(img2.screen,"Images/"+type(agent).__name__+"_bonus"+str(trial)+".png")
                    if type(agent).__name__=='Rmax_Agent':
                        reward_bonus=copy.deepcopy(agent.R)
                        img2=picture_world(environment,reward_bonus)
                        pygame.image.save(img2.screen,"Images/"+type(agent).__name__+"_bonus"+str(trial)+".png")
                    if type(agent).__name__=='BEB_Agent':
                        beta_bonus=copy.deepcopy(agent.bonus)
                        img2=picture_world(environment,beta_bonus)
                        pygame.image.save(img2.screen,"Images/"+type(agent).__name__+"_bonus"+str(trial)+".png")


def normalized_table(table,environment):
    max_every_state=np.zeros((environment.height,environment.width))
    for state in table.keys():
        q_values = table[state]
        max_value_state=max(q_values.values())
        max_every_state[state]=max_value_state
    mini,maxi=np.min(max_every_state),np.max(max_every_state)
    if mini < 0 : max_every_state-=mini
    if maxi !=0 : max_every_state/=maxi     
    return max_every_state
 
def picture_world(environment,table):       
    precision=normalized_table(table,environment)
    init_loc=[environment.first_location[0],environment.first_location[1]]
    screen_size = environment.height*50
    cell_width = 44.8
    cell_height = 44.8
    cell_margin = 5
    gridworld = Graphique(screen_size,cell_width, cell_height, cell_margin,environment.grid,environment.final_states,init_loc,precision)
    return gridworld



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
    
#Stats 


def convergence(array,longueur=20,variation=0.2,absolu=3,artefact=3):
    for i in range(len(array)-longueur):
        table=array[i:i+longueur]
        mean=np.mean(table)
        if mean >0:
            mauvais=0
            valid=True
            for element in table : 
                if element < (1-variation)*mean-absolu>element or element > (1+variation)*mean+absolu:
                    mauvais+=1
                    if mauvais > artefact:
                        valid=False
            if valid : 
                return [len(array)-len(array[i+1:]),np.mean(array[i+1:]),np.var(array[i+1:])]            
    return [len(array)-1,np.mean(array),np.var(array)]
    
    
    
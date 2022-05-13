import numpy as np
import copy
import pygame
import seaborn as sns
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

from Gridworld import State
from Complexworld import ComplexState
from Deterministic_world import Deterministic_State
from Uncertain_world import Uncertain_State
from Lopesworld import Lopes_State
from Two_step_task import Two_step
from Lopes_nonstat import Lopes_nostat
from Deterministic_nostat import Deterministic_no_stat
from Uncertainworld_U import Uncertain_State_U
from Uncertainworld_B import Uncertain_State_B

from Q_learning import Q_Agent
from Kalman import Kalman_agent
from Kalman_sum import Kalman_agent_sum
from Rmax import Rmax_Agent
from BEB import BEB_Agent
from BEBLP import BEBLP_Agent
from RmaxLP import RmaxLP_Agent

from Representation import Graphique



#MAIN

def play(environment, agent, trials=200, max_step=500, screen=1,photos=[10,20,50,100,199,300,499],accuracy=0.01,pas_VI=100):
    reward_per_episode = []
    step_number=[]
    policy_value_error=[]
    pol_updated=False
    val_iteration,_=value_iteration(environment,agent.gamma,accuracy)
    for trial in range(trials):
        if screen : take_picture(agent,trial,environment,photos) #Visualisation
        
        cumulative_reward, step, game_over= 0,0,False
        while not game_over :
            if type(environment).__name__ in ['Uncertain_State_U','Uncertain_State_B','Deterministic_no_stat','Lopes_nostat'] and not pol_updated:
                if environment.changed :
                    pol_updated=True
                    val_iteration,_=value_iteration(environment,agent.gamma,accuracy)
            if agent.step_counter%pas_VI==0:
                policy_value_error.append(policy_evaluation(environment,get_policy(agent),agent.gamma,accuracy)[environment.first_location]-val_iteration[environment.first_location]) 
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
    return reward_per_episode,step_number,policy_value_error

### Initializing environments ###

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


### PICTURES ###

def take_picture(agent,trial,environment,photos):
            if trial in photos:
                    value=copy.deepcopy(agent.Q)
                    img=picture_world(environment,value)
                    if type(agent).__name__ in ['BEB_Agent','Rmax_Agent','Kalman_agent_sum','BEBLP_Agent','RmaxLP_Agent']:
                        pygame.image.save(img.screen,"Images/Solo/"+type(agent).__name__+"_"+type(environment).__name__+"_"+str(trial)+".png")
                    else : pygame.image.save(img.screen,"Images/"+type(agent).__name__+"_"+type(environment).__name__+"_"+str(trial)+".png")
                    if type(agent).__name__ =='Kalman_agent_sum': bonus=copy.deepcopy(agent.K_var)
                    if type(agent).__name__=='Rmax_Agent': bonus=copy.deepcopy(agent.R)
                    if type(agent).__name__=='BEB_Agent': bonus=copy.deepcopy(agent.bonus)
                    if type(agent).__name__=='BEBLP_Agent': bonus=copy.deepcopy(agent.bonus)
                    if type(agent).__name__=='RmaxLP_Agent': bonus=copy.deepcopy(agent.R)
                    if type(agent).__name__ in ['BEB_Agent','Rmax_Agent','Kalman_agent_sum','BEBLP_Agent','RmaxLP_Agent']:
                        img2=picture_world(environment,bonus)
                        pygame.image.save(img2.screen,"Images/Solo/"+type(agent).__name__+"_bonus"+"_"+type(environment).__name__+str(trial)+".png")
                        merging_two_images(environment,"Images/Solo/"+type(agent).__name__+"_"+type(environment).__name__+"_"+str(trial)+".png","Images/Solo/"+type(agent).__name__+"_bonus"+"_"+type(environment).__name__+str(trial)+".png","Images/"+type(agent).__name__+"_"+type(environment).__name__+" Q_table (left) and bonus (right) "+str(trial)+".png")

def merging_two_images(environment,img1,img2,path):
    pygame.init()
    image1 = pygame.image.load(img1)
    image2 = pygame.image.load(img2)

    screen = pygame.Surface((environment.height*100+200,environment.height*50+100))
    
    screen.fill((0,0,0))   
    screen.blit(image1, (50,  50))
    screen.blit(image2, (environment.height*50+150, 50))

    pygame.image.save(screen,path)

    
        
def normalized_table(table,environment):
    max_every_state=np.zeros((environment.height,environment.width))
    action_every_state=dict()
    for state in table.keys():
        q_values = table[state]
        max_value_state=max(q_values.values())
        max_every_state[state]=max_value_state
        
        best_action = np.random.choice([k for k, v in q_values.items() if v == max_value_state])
        action_every_state[state]=best_action
    mini,maxi=np.min(max_every_state),np.max(max_every_state)
    if mini < 0 : max_every_state-=mini
    if maxi !=0 : max_every_state/=maxi     
    return max_every_state,action_every_state
 
def picture_world(environment,table):       
    max_Q,best_actions=normalized_table(table,environment)
    init_loc=[environment.first_location[0],environment.first_location[1]]
    screen_size = environment.height*50
    cell_width = 45
    cell_height = 45
    cell_margin = 5
    gridworld = Graphique(screen_size,cell_width, cell_height, cell_margin,environment.grid,environment.final_states,init_loc,max_Q,best_actions)
    return gridworld

### SAVING PARAMETERS ####

def save_pickle(dictionnaire,path):
    with open(path, 'wb') as f:
        pickle.dump(dictionnaire, f) 

def open_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

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

def convergence(array,longueur=30,variation=0.2,absolu=0.1,artefact=3):
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

#Evaluating a policy

def value_iteration(environment,gamma,accuracy):
    V={state:1 for state in environment.states}
    action={state:0 for state in environment.states}
    delta=accuracy+1
    while delta > accuracy :
        delta=0
        for state,value in V.items():
            value_V=V[state]
            V[state]=np.max([np.sum([environment.transitions[action][state][new_state]*(environment.values[state[0],state[1],action]+gamma*V[new_state]) for new_state in environment.transitions[action][state].keys()]) for action in environment.actions])
            delta=max(delta,np.abs(value_V-V[state]))
            action[state]=np.argmax([np.sum([environment.transitions[action][state][new_state]*(environment.values[state[0],state[1],action]+gamma*V[new_state]) for new_state in environment.transitions[action][state].keys()]) for action in environment.actions])
    return V,action

def plot_VI(environment,gamma,accuracy): #only in gridworlds
    V,action=value_iteration(environment,gamma,accuracy)
    V_2=np.zeros((environment.height,environment.width))
    for state,value in V.items():
        V_2[state]=value
        if V_2[state]<0:V_2[state]=0
    max_value=np.max(V_2)
    V_2=V_2/max_value
    init_loc=[environment.first_location[0],environment.first_location[1]]
    screen_size = environment.height*50
    cell_width = 45
    cell_height = 45
    cell_margin = 5
    gridworld = Graphique(screen_size,cell_width, cell_height, cell_margin,environment.grid,environment.final_states,init_loc,V_2,action)
    #pygame.image.save(gridworld.screen,"Images/Optimal policy/VI_"+type(environment).__name__+".png")
    return gridworld

def policy_evaluation(environment,policy,gamma,accuracy):
    V={state:1 for state in environment.states}
    delta=accuracy+1
    for state in V.keys():
        if state not in policy.keys():
            policy[state]=np.random.choice(environment.actions)
    while delta > accuracy :
        delta=0
        for state,value in V.items():
            value_V=V[state]
            action=policy[state]
            V[state]=np.sum([environment.transitions[action][state][new_state]*(environment.values[state[0],state[1],action]+gamma*V[new_state]) for new_state in environment.transitions[action][state].keys()])
            delta=max(delta,np.abs(value_V-V[state]))
    return V

def get_policy(agent):
    action_every_state=dict()
    for state in agent.Q.keys():
        q_values = agent.Q[state]
        max_value_state=max(q_values.values())
        best_action = np.random.choice([k for k, v in q_values.items() if v == max_value_state])
        action_every_state[state]=best_action 
    return action_every_state

##Optimal policies ###

def compute_optimal_policies(environments_parameters=environments_parameters):
    for name_environment in all_environments.keys():
        environment=all_environments[name_environment](**environments_parameters[name_environment])
        for number_world in range(1,21):
            world=np.load('Mondes/World_'+str(number_world)+'.npy')
            world_2=np.load('Mondes/World_'+str(number_world)+'_B.npy')
            transitions=np.load('Mondes/Transitions_'+str(number_world)+'.npy',allow_pickle=True)
            transitions_U=np.load('Mondes/Transitions_'+str(number_world)+'_U.npy',allow_pickle=True)
            transitions_B=np.load('Mondes/Transitions_'+str(number_world)+'_B.npy',allow_pickle=True)
            transitions_lopes=np.load('Mondes/Transitions_Lopes_non_stat'+str(number_world)+'.npy',allow_pickle=True)
            environments_parameters["DB_{0}".format(number_world)] = {'world':world_2,'world2':world}
            environments_parameters["UU_{0}".format(number_world)] = {'world':world,'transitions':transitions_U,'transitions_U':transitions}
            environments_parameters["UB_{0}".format(number_world)] = {'world':world_2,'world2':world,'transitions':transitions_B,'transitions_B':transitions} 
            environments_parameters["Lopes_nostat_{0}".format(number_world)]={'transitions':transitions_lopes,'transitions2':transitions}
        if type(environment).__name__ !='Two_step':
            gridworld=plot_VI(environment,gamma=0.95,accuracy=0.001)
            pygame.image.save(gridworld.screen,"Images/Optimal policy/VI_"+name_environment+".png")

### Parametter fitting ##

environment_names=['Lopes']
betas=[i for i in range(1,2)]
priors=[1*i for i in range(1,100,10)]
def fitting_BEB(environment_names,betas,priors,trials = 300,max_step = 30,accuracy=5,screen=0):
    BEB_parameters={(beta,prior):{'gamma':0.95,'beta':beta,'known_states':True,'coeff_prior':prior} for beta in betas for prior in priors}
    pol_error={}
    for name_environment in environment_names:   
        print(name_environment)
        environment=all_environments[name_environment](**environments_parameters[name_environment])                
        for beta in betas :
            print(beta)
            for prior in priors :
                BEB=BEB_Agent(environment,**BEB_parameters[(beta,prior)]) #Defining a new agent from the dictionary agents
                
                reward,step_number,policy_value_error= play(environment,BEB,trials=trials,max_step=max_step,screen=screen,accuracy=accuracy) #Playing in environment
                index_convergence=-1
                for i in range(len(policy_value_error)-1):
                    if policy_value_error[i] >-0.5 and policy_value_error[i+1] >-0.5 :
                        index_convergence=i*50
                        break
                pol_error[beta,prior]=index_convergence
    return pol_error


alphas=[0.1*i for i in range(1,10,1)]
ms=[0.1*i for i in range(1,17,2)]
def fitting_RALP(environment_names,alphas,ms,trials = 200,max_step = 30,accuracy=.05,screen=0):
    RALP_parameters={(alpha,m):{'gamma':0.95,'Rmax':1,'step_update':10,'alpha':alpha,'m':m,'VI':50} for alpha in alphas for m in ms}
    pol_error={}
    for name_environment in environment_names:   
        print(name_environment)
        environment=all_environments[name_environment](**environments_parameters[name_environment])                
        for alpha in alphas :
            print(alpha)
            for m in ms :
                RALP=RmaxLP_Agent(environment,**RALP_parameters[(alpha,m)]) #Defining a new agent from the dictionary agents
                
                reward,step_number,policy_value_error= play(environment,RALP,trials=trials,max_step=max_step,screen=screen,accuracy=accuracy) #Playing in environment
                index_convergence=-1
                for i in range(len(policy_value_error)-1):
                    if policy_value_error[i] >-0.5 and policy_value_error[i+1] >-0.5 :
                        index_convergence=i*50
                        break
                pol_error[alpha,m]=index_convergence
    return pol_error



#a=fitting_BEB(environment_names,betas,priors)




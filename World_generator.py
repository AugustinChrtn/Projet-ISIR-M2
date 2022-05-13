import numpy as np 
import time
import pygame
def create_matrix(width,height,liste):
    return [[liste for i in range(width)] for j in range(height)]
import random
from Monde_representation import Monde
from Representation import Transition
from Lopesworld import Lopes_State
import copy
from collections import defaultdict

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4


pattern={(i,j):0 for i in range(4,7) for j in range(4,7)}
for i in [4,6]:
    for j in [4,6]:
        pattern[i,j]=-1
taille=10
pourcent_murs=0.1

def monde_avec_murs(taille=taille,pourcent_murs=pourcent_murs,pattern=pattern):
    etats=np.zeros((taille,taille))
    #1ere phase
    for i in range(taille):
        for j in range(taille):
            if random.random()<pourcent_murs:
                etats[i,j]=-1        
    #2e phase 
    for i in range(taille):
        for j in range(taille):
            r=0
            if i-1>0 and etats[i-1,j]==-1:r+=0.25
            if i+1< taille-1 and etats[i+1,j]==-1:r+=0.25
            if j-1>0 and etats[i,j-1]==-1:r+=0.25
            if j+1< taille-1 and etats[i,j+1]==-1:r+=0.25
            if random.random()<r:etats[i,j]=-1
    #3e phase         
    for i in range(taille):
        for j in range(taille):
            r=0
            if i-1<0 or etats[i-1,j]==-1:r+=0.25
            if i+1> taille-1 or etats[i+1,j]==-1:r+=0.25
            if j-1<0 or etats[i,j-1]==-1:r+=0.25
            if j+1 > taille-1 or etats[i,j+1]==-1:r+=0.25
            if r==1:etats[i,j]=-1
    #pattern 
    for case,value in pattern.items() :
        etats[case]=value
    return etats

def etat_initial(etats):
    etats_init=[]
    for i in range(len(etats)):
        for j in range(len(etats)):
            if etats[i,j]==0:etats_init.append((i,j))
    etats_init=np.array(etats_init)
    indices = np.arange(etats_init.shape[0])
    etat_initial= etats_init[np.random.choice(indices)]
    etats[etat_initial[0],etat_initial[1]]=1
    return etats

def distance_etat_initial(etats):
    for k in range(1,50):
        for i in range(len(etats)):
            for j in range(len(etats[i])):
                if etats[i][j] == k:
                    if i>0 and etats[i-1][j] == 0:
                        etats[i-1][j] = k + 1
                    if j>0 and etats[i][j-1] == 0 :
                        etats[i][j-1] = k + 1
                    if i<len(etats)-1 and etats[i+1][j] == 0 :
                        etats[i+1][j] = k + 1
                    if j<len(etats[i])-1 and etats[i][j+1] == 0 :
                        etats[i][j+1] = k + 1
    return etats

def generation_distance(taille=taille,pourcent_murs=pourcent_murs):
    monde=monde_avec_murs(taille,pourcent_murs)
    monde_initial=etat_initial(monde)
    monde_distance=distance_etat_initial(monde_initial)
    #Monde valide?
    high_rewards=(monde_distance>15).any()
    return monde_distance,high_rewards

recompenses=[0.2,1]

def generer_un_monde(taille=taille,pourcent_murs=pourcent_murs,recompenses=recompenses,pattern=pattern):
    valid=False
    while not valid:
        monde,valid=generation_distance()
    distance_initial=monde.copy()
    more_than_15=[]
    between_5_and_10=[]
    for row in range(taille):
        for col in range(taille):
            if monde[row,col]>15:more_than_15.append((row,col))
            if monde[row,col]>5 and monde[row,col]<=10:between_5_and_10.append((row,col))
    more_than_15,between_5_and_10=np.array(more_than_15),np.array(between_5_and_10)
    indices_1,indices_2=np.arange(more_than_15.shape[0]),np.arange(between_5_and_10.shape[0])
    high_reward,low_reward=more_than_15[np.random.choice(indices_1)],between_5_and_10[np.random.choice(indices_2)]
    for row in range(taille):
        for col in range(taille):
            if monde[row,col] not in [-1,1]: monde[row,col]=0
            elif monde[row,col]==1: monde[row,col]=-2
            monde[high_reward[0],high_reward[1]]=np.max(recompenses)
            monde[low_reward[0],low_reward[1]]=np.min(recompenses)
            
    
    #distance from the max reward
    
    monde_highest_reward=monde.copy()
    for row in range(taille):
        for col in range(taille):
            if monde_highest_reward[row,col]!=-1:
                monde_highest_reward[row,col]=0
    monde_highest_reward[high_reward[0],high_reward[1]]=1
    distance_max=distance_etat_initial(monde_highest_reward)

    #Cases on an optimal path
    optimal_path=np.zeros((taille,taille))
    for row in range(taille):
        for col in range(taille):
            if distance_initial[row,col] >=1  and distance_max[row,col]>=1:
                optimal_path[row,col]= distance_initial[row,col]+distance_max[row,col]<=distance_initial[high_reward[0],high_reward[1]]+1
            else : optimal_path[row,col]=-1
    
    distance_init_optimal=dict()
    for row in range(taille):
        for col in range(taille):
            if optimal_path[row,col]==1 and (row,col) not in pattern.keys():
                distance_init_optimal[row,col]=distance_initial[row,col]
    
    monde_valide=False
    
    for case,value in pattern.items() :
        if value ==0:
            if distance_initial[case[0],case[1]] > 6 and distance_max[case[0],case[1]]>6 : #Far from the max reward
                #Pattern on the optimal way
                if optimal_path[case] and len([k for k, v in distance_init_optimal.items() if v == distance_initial[case[0],case[1]]])==0: monde_valide=True
        if (low_reward[0],low_reward[1]) in pattern.keys(): #Low reward not in the pattern
            monde_valide=False
    

    monde_2=monde.copy()
    for case,value in pattern.items():
        monde_2[case]=-1
    monde_2[high_reward[0],high_reward[1]]=0
    monde_2[low_reward[0],low_reward[1]]=0
    for row in range(taille):
        for col in range(taille):
            if monde_2[row,col]==-2:
                monde_2[row,col]=1
    distance_monde_2=distance_etat_initial(monde_2)
    if distance_monde_2[high_reward[0],high_reward[1]] <= distance_initial[high_reward[0],high_reward[1]]+3 : monde_valide=False
    if not monde_valide : return generer_un_monde(taille=taille,pourcent_murs=pourcent_murs,recompenses=recompenses,pattern=pattern)
    return monde

exemple=generer_un_monde()
gridworld=Monde(exemple,recompenses)
pygame.display.flip()
pygame.time.delay(1000)
pygame.quit()   
    
def generer_des_mondes(nombre=20):
    for i in range(nombre):
        monde=generer_un_monde()
        np.save('Mondes/World_'+str(i+1)+'.npy',monde)
        gridworld=Monde(monde,recompenses)
        pygame.display.flip()
        pygame.image.save(gridworld.screen,"Mondes/World_"+str(i+1)+".png")
        pygame.quit()

def generer_des_mondes_bloques(nombre=20):
    for i in range(20):
        world=np.load('Mondes/World_' + str(i+1) +'.npy')
        for row in range(len(world)):
            for col in range(len(world)):
                if (row,col) in pattern.keys():
                    world[row,col]=-1
        np.save('Mondes/World_'+str(i+1)+'_B.npy',world)
        gridworld=Monde(world,recompenses)
        pygame.display.flip()
        pygame.image.save(gridworld.screen,"Mondes/World_"+str(i+1)+"_R.png")
        pygame.quit()
        for row in range(len(world)):
            for col in range(len(world)):
                if world[row][col] not in [-1,-2]: world[row,col]=0
                if world[row][col]==-2: world[row,col]=1
        monde_distance=distance_etat_initial(world)
        gridworld=Monde(monde_distance)
        pygame.display.flip()
        pygame.image.save(gridworld.screen,"Mondes/World_"+str(i+1)+"_distance_B.png")
        pygame.quit()
                

def incertitude_transition(world):
   walls=[]
   for row in range(len(world)):
        for col in range(len(world[0])):
            if world[row][col]==-1: walls.append((row,col))
   dict_transitions=[[list({} for i in range(len(world)))for i in range(len(world[0]))] for i in range(5)]
   for row in range(len(world)):
       for col in range(len(world[0])):
           if world[row][col] !=-1:
               for action in range(4):
                   
                   uncertainty=random.randint(20,50)*0.01
                   bias=random.randint(10,90)*0.01
                   spread_left=random.randint(0,20)*0.01
                   spread_right=random.randint(0,20)*0.01
                   stay=random.randint(10,20)*0.01
                   deterministic=1-uncertainty
                   proba_stay=stay*uncertainty
                   proba_right=uncertainty*(1-stay)*bias
                   proba_left=uncertainty*(1-stay)*(1-bias)
                   proba_left1=proba_left*(1-spread_left)
                   proba_left2=proba_left*spread_left
                   proba_right1=proba_right*(1-spread_right)
                   proba_right2=proba_right*spread_right
                   
                   probas=np.array([proba_left2,proba_left1,deterministic,proba_right1,proba_right2,proba_stay])
                   
                   
                   row_0, row_10, col_0, col_10 = row==0,row==len(world)-1,col==0,col==len(world)-1
                   limites=[(row_0,col_0,col_10), (row_10,col_10,col_0), (col_0,row_10,row_0),(col_10,row_0,row_10)]
                   if action == UP : 
                       cases=[(row,col-1),(row-1,col-1),(row-1,col),(row-1,col+1),(row,col+1),(row,col)]
                   elif action == DOWN : 
                       cases =[(row,col+1),(row+1,col+1),(row+1,col),(row+1,col-1),(row,col-1),(row,col)]
                   elif action == LEFT :
                       cases =[(row+1,col),(row+1,col-1),(row,col-1),(row-1,col-1),(row-1,col),(row,col)]
                   elif action == RIGHT : 
                       cases = [(row-1,col),(row-1,col+1),(row,col+1),(row+1,col+1),(row+1,col),(row,col)]
                       
                   if limites[action][0] : 
                           probas[5]+=probas[1]+probas[2]+probas[3]
                           probas[1],probas[2],probas[3]=0,0,0
                   if cases[2] in walls : 
                               probas[5]+=probas[2]
                               probas[2]=0
                   if limites[action][1] :
                           probas[5]+=probas[0]+probas[1]
                           probas[0],probas[1]=0,0
                   if cases[1] in walls : 
                               probas[5]+=probas[1]
                               probas[1]=0
                   if cases[0] in walls :
                               probas[5]+=probas[0]
                               probas[0]=0
                   if limites[action][2]:
                               probas[5]+=probas[3]+probas[4]
                               probas[3],probas[4]=0,0
                   if cases[3] in walls : 
                               probas[5]+=probas[3]
                               probas[3]=0
                   if cases[4] in walls : 
                               probas[5]+=probas[4]
                               probas[4]=0
                   if cases[2] in walls :
                           if cases[0] in walls :
                               probas[5]+=probas[1]
                               probas[1]=0
                           if cases[4] in walls : 
                               probas[5]+=probas[3]
                               probas[3]=0
                   probas=np.round(probas,5)
                   for i in range(6):
                           if probas[i]!=0:
                               dict_transitions[action][row][col][cases[i]]=probas[i]
               dict_transitions[4][row][col][(row,col)]=1
   return dict_transitions
                           
def generer_des_mondes_incertains(nombre=20):
    for i in range(1,21):
        world=np.load('Mondes/World_' + str(i)+'.npy')
        transitions=incertitude_transition(world)
        np.save('Mondes/Transitions_'+str(i)+'.npy',transitions)
        
def generer_distance_monde(nombre=20):
    for i in range(20):
        world=np.load('Mondes/World_' + str(i+1) +'.npy')
        for row in range(len(world)):
            for col in range(len(world)):
                if world[row][col] not in [-1,-2]: world[row,col]=0
                if world[row][col]==-2: world[row,col]=1
        monde_distance=distance_etat_initial(world)
        gridworld=Monde(monde_distance)
        pygame.display.flip()
        pygame.image.save(gridworld.screen,"Mondes/World_"+str(i+1)+"_distance"+".png")
        pygame.quit()

def montrer_transition(world_number,action,row,col):
    world=np.load('Mondes/World_' + str(world_number) +'.npy')
    all_transitions=np.load('Mondes/Transitions_'+str(world_number)+'.npy',allow_pickle=True)
    transition=all_transitions[action][row][col]
    walls={(row-1,col-1):0,(row-1,col):0,(row-1,col+1):0,(row,col-1):0,(row,col):0,(row,col+1):0,(row+1,col-1):0,(row+1,col):0,(row+1,col+1):0}
    for case in walls.keys() : 
        if case[0]<0 or case[1]<0 or case[0]>=len(world) or case[1]>=len(world) or world[case[0],case[1]]==-1:
            walls[case]=1
    actions=['UP','DOWN','LEFT','RIGHT']
    titre ="Action " +actions[action]+ ", case ("+str(row)+","+str(col)+")"+" dans le monde "+str(world_number)
    transi=Transition((row,col),titre,transition,walls,action)
    pygame.display.flip()
    pygame.image.save(transi.screen,"Mondes/"+str(titre)+'.png')
    pygame.time.delay(5000)
    pygame.quit()
    
def convert_from_default(dic):
    from collections import defaultdict
    if isinstance(dic,defaultdict):
        return dict((key,convert_from_default(value)) for key,value in dic.items())
    else : return dic
    
def generer_pattern_incertain(nombre=20):
    for i in range(nombre):
        transitions=np.load('Mondes/Transitions_' + str(i+1) +'.npy',allow_pickle=True)
        for (row,col),value in pattern.items():
            if value==0:
                new_transitions=defaultdict(lambda:0.0)
                for action in range(5):
                    for key,value in transitions[action][row,col].items():
                        new_transitions[key]+=value
                new_transitions=convert_from_default(new_transitions)
                final_transitions={k:v/5 for k,v in new_transitions.items()}
                for action in range(5):
                    transitions[action][row,col]=final_transitions
        np.save('Mondes/Transitions_'+str(i+1)+'_U.npy',transitions)
        
def generer_transitions_bloquees(nombre=20):
    for i in range(nombre):
        transitions=np.load('Mondes/Transitions_' + str(i+1) +'.npy',allow_pickle=True)
        transitions_2=copy.deepcopy(transitions)
        for row in range(taille):
            for col in range(taille):
                for action in range(5):
                    new_transitions={(row,col):0}
                    for key,value in transitions[action][row,col].items():
                        if key in pattern.keys() or key==(row,col):
                                new_transitions[row,col]+=value
                        else : new_transitions[key]=value
                    transitions_2[action][row,col]=new_transitions
        for (row,col),value in pattern.items():
                for action in range(5):
                    transitions_2[action][row,col]={}
        np.save('Mondes/Transitions_' + str(i+1) +'_B.npy',transitions_2)
        
def tout_generer(nombre=20):
    generer_des_mondes(nombre)
    generer_des_mondes_incertains(nombre)
    generer_distance_monde(nombre)
    generer_pattern_incertain(nombre)
    generer_transitions_bloquees(nombre)
    
###---------------LOPES-----------###       
    
def transition_Lopes():
        dict_transitions=[[list({} for i in range(5))for i in range(5)] for i in range(5)]
        uncertain_states=[(0,1),(0,3),(2,1),(2,3)]
        for action in [UP,DOWN,LEFT,RIGHT,STAY]:
            for height in range(5):
                for width in range(5):
                    if (height,width) not in uncertain_states:probas=np.random.dirichlet([0.1]*25)
                    else : probas = np.random.dirichlet([1]*25)
                    probas=probas.reshape(5,5)
                    if action == UP and height-1 >=0: index=(height-1,width)
                    elif action == DOWN and height+1 <5: index = (height+1,width)
                    elif action == LEFT and width-1 >=0: index = (height,width-1)
                    elif action == RIGHT and width+1 <5: index = (height,width+1)
                    else : index=(height,width) 
                    max_index=np.unravel_index(probas.argmax(),probas.shape)
                    probas[max_index],probas[index]=probas[index],probas[max_index]
                    for row in range(5):
                        for col in range(5):
                            dict_transitions[action][height][width][(row,col)]=probas[row][col]
        np.save('Mondes/Transitions_Lopes.npy',dict_transitions)
        
def transition_Lopes_2():
        dict_transitions=[[list({} for i in range(5))for i in range(5)] for i in range(5)]
        uncertain_states=[(0,1),(0,3),(2,1),(2,3)]
        for action in [UP,DOWN,LEFT,RIGHT,STAY]:
            for height in range(5):
                for width in range(5):
                    if (height,width) not in uncertain_states:alpha=0.1
                    else : alpha=1
                    if action == UP and height-1 >=0: ind=(height-1,width)
                    elif action == DOWN and height+1 <5: ind = (height+1,width)
                    elif action == LEFT and width-1 >=0: ind = (height,width-1)
                    elif action == RIGHT and width+1 <5: ind = (height,width+1)
                    else : ind=(height,width) 
                    etats=[(height,width), (height-1,width),(height+1,width),(height,width-1),(height,width+1)]
                    if height-1 <0: etats.remove((height-1,width))
                    if height+1 ==5: etats.remove((height+1,width))
                    if width-1 <0:  etats.remove((height,width-1))
                    if width+1==5:  etats.remove((height,width+1))
                    values=np.random.dirichlet([alpha]*len(etats))
                    probas={etats[i]:values[i] for i in range(len(etats))}
                    maxValue = max(probas.values())
                    max_ind = [k for k, v in probas.items() if v == maxValue]
                    j=np.random.randint(len(max_ind))                   
                    probas[max_ind[j]],probas[ind]=probas[ind],probas[max_ind[j]]
                    for key in probas.keys() :
                            dict_transitions[action][height][width][key]=probas[key]
        np.save('Mondes/Transitions_Lopes.npy',dict_transitions)
        
        
def transition_Lopes_3():
        dict_transitions=[[list({} for i in range(5))for i in range(5)] for i in range(5)]
        uncertain_states=[(0,1),(0,3),(2,1),(2,3)]
        for action in [UP,DOWN,LEFT,RIGHT,STAY]:
            for height in range(5):
                for width in range(5):
                    if (height,width) not in uncertain_states:alpha=0.1
                    else : alpha=1
                    if action == UP and height-1 >=0: ind=(height-1,width)
                    elif action == DOWN and height+1 <5: ind = (height+1,width)
                    elif action == LEFT and width-1 >=0: ind = (height,width-1)
                    elif action == RIGHT and width+1 <5: ind = (height,width+1)
                    else : ind=(height,width) 
                    etats=[(height,width), (height-1,width),(height+1,width),(height,width-1),(height,width+1)]
                    values=np.random.dirichlet([alpha]*5)
                    values=np.random.dirichlet([alpha]*len(etats))
                    probas={etats[i]:values[i] for i in range(len(etats))}
                    maxValue = max(probas.values())
                    max_ind = [k for k, v in probas.items() if v == maxValue]
                    j=np.random.randint(len(max_ind))                   
                    probas[max_ind[j]],probas[ind]=probas[ind],probas[max_ind[j]]
                    if height-1 <0: 
                        probas[(height,width)]+=probas[(height-1,width)]
                        del probas[(height-1,width)]
                    if height+1 ==5:
                        probas[(height,width)]+=probas[(height+1,width)]
                        del probas[(height+1,width)]
                    if width-1 <0:
                        probas[(height,width)]+=probas[(height,width-1)]
                        del probas[(height,width-1)]
                    if width+1==5:
                        probas[(height,width)]+=probas[(height,width+1)]
                        del probas[(height,width+1)]
                    for key in probas.keys() :
                            dict_transitions[action][height][width][key]=probas[key]
        np.save('Mondes/Transitions_Lopes.npy',dict_transitions)

def non_stat_Lopes_article(nombre=20):
    for i in range(nombre):
        transitions=np.load('Mondes/Transitions_Lopes.npy',allow_pickle=True)
        optimal_path=[(0,0),(1,0),(2,0),(3,0),(3,1),(3,2),(3,3),(3,4),(2,4)]
        
        index_changed=np.random.randint(len(optimal_path))
        state_to_change=optimal_path[index_changed]
        liste_rotation=[j for j in range(5)]
        valid=False
        while not valid:
            valid=True
            for k in range(5):
                if liste_rotation[k]==k:
                    valid=False
                    random.shuffle(liste_rotation)
                    break
        new_transitions=[transitions[rotation][state_to_change[0]][state_to_change[1]] for rotation in liste_rotation]
        for action in range(5):
            transitions[action][state_to_change[0]][state_to_change[1]]=new_transitions[action]
        print(state_to_change)
        np.save('Mondes/Transitions_Lopes_non_stat'+str(i+1)+'.npy',transitions)
        
        
def non_stat_Lopes_optimal(nombre=20):
    for i in range(nombre):
        transitions=np.load('Mondes/Transitions_Lopes.npy',allow_pickle=True)
        optimal_path=[(0,0),(1,0),(2,0),(3,0),(3,1),(3,2),(3,3),(3,4),(2,4)]
        
        for state_to_change in optimal_path:
            liste_rotation=[j for j in range(5)]
            valid=False
            while not valid:
                valid=True
                for k in range(5):
                    if liste_rotation[k]==k:
                        valid=False
                        random.shuffle(liste_rotation)
                        break
            new_transitions=[transitions[rotation][state_to_change[0]][state_to_change[1]] for rotation in liste_rotation]
            for action in range(5):
                transitions[action][state_to_change[0]][state_to_change[1]]=new_transitions[action]
        np.save('Mondes/Transitions_Lopes_non_stat'+str(i+1)+'.npy',transitions)
        
        
def non_stat_Lopes(nombre=20):
    for i in range(nombre):
        transitions=np.load('Mondes/Transitions_Lopes.npy',allow_pickle=True)
        all_states=[(i,j) for i in range(5) for j in range(5)]
        
        for state_to_change in all_states:
            liste_rotation=[j for j in range(5)]
            random.shuffle(liste_rotation)
            new_transitions=[transitions[rotation][state_to_change[0]][state_to_change[1]] for rotation in liste_rotation]
            for action in range(5):
                transitions[action][state_to_change[0]][state_to_change[1]]=new_transitions[action]
        np.save('Mondes/Transitions_Lopes_non_stat'+str(i+1)+'.npy',transitions)




    
"""from Useful_functions import value_iteration

def valid_Lopes():
    valid=False
    count=0
    while not valid and count <500: 
        print(count)
        transitions=np.load('Mondes/Transitions_Lopes.npy',allow_pickle=True)
        environment=Lopes_State(transitions)
        _,policy=value_iteration(environment,0.95,0.01)
        if policy[0,0]==1 and policy[1,0]==1 and policy[2,0]==1 and policy[3,0]==3 and policy[3,1]==3 and policy[3,2]==3 and policy[3,3]==3 and policy[3,4]==0 and policy[2,4]==4:
            valid =True
        else : 
            count+=1
            transition_Lopes_2()"""
            
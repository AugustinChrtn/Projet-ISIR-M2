import numpy as np 
import time
import pygame
def create_matrix(width,height,liste):
    return [[liste for i in range(width)] for j in range(height)]
import random
from Monde_representation import Monde
from Representation import Transition

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4

def monde_avec_murs(taille=10,pourcent_murs=0.1):
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

def generation_distance(taille=10,pourcent_murs=0.1):
    monde=monde_avec_murs(taille,pourcent_murs)
    monde_initial=etat_initial(monde)
    monde_distance=distance_etat_initial(monde_initial)
    #Monde valide?
    high_rewards=(monde_distance>15).any()
    return monde_distance,high_rewards

def generer_un_monde(taille=10,pourcent_murs=0.1,recompenses=[20,200]):
    valid=False
    while not valid:
        monde,valid=generation_distance()
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
            if monde[row,col] not in [-1,1]:
                monde[row,col]=0
            monde[high_reward[0],high_reward[1]]=np.max(recompenses)
            monde[low_reward[0],low_reward[1]]=np.min(recompenses)
    return monde

"""exemple=generer_un_monde()
gridworld=Monde(exemple)
pygame.display.flip()
pygame.quit()"""            
    
def generer_des_mondes(nombre=20):
    for i in range(nombre):
        monde=generer_un_monde()
        np.save('Mondes/World_'+str(i+1)+str(int(time.time()))+'.npy',monde)
        gridworld=Monde(monde)
        pygame.display.flip()
        pygame.image.save(gridworld.screen,"Mondes/World_"+str(i+1)+str(int(time.time()))+".png")
        pygame.quit()

def incertitude_transition(world):
   walls=[]
   for row in range(len(world)):
        for col in range(len(world[0])):
            if world[row][col]==-1: walls.append((row,col))
   dict_transitions=[[list({} for i in range(len(world)))for i in range(len(world[0]))] for i in range(4)]
   for row in range(len(world)):
       for col in range(len(world[0])):
           if world[row][col] !=-1:
               for action in range(4):
                   
                   uncertainty=random.randint(10,30)*0.01
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
                               
   return dict_transitions
                           
def generer_des_mondes_incertains(nombre=20):
    for i in range(1,21):
        world=np.load('Mondes/World_' + str(i) +'.npy')
        transitions=incertitude_transition(world)
        np.save('Mondes/Transitions_'+str(i)+'.npy',transitions)
        
def generer_distance_monde(nombre=20):
    for i in range(20):
        world=np.load('Mondes/World_' + str(i+1) +'.npy')
        for row in range(len(world)):
            for col in range(len(world)):
                if world[row][col]==20 : world[row][col]=0
                if world[row][col]==200 : world[row][col]=0
        monde_distance=distance_etat_initial(world)
        gridworld=Monde(monde_distance)
        pygame.display.flip()
        pygame.image.save(gridworld.screen,"Mondes/World_"+str(i+1)+"_distance"+str(int(time.time()))+".png")
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
    
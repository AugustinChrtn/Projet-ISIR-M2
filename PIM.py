import numpy as np
from collections import defaultdict
import random

def count_to_dirichlet(dictionnaire):
    keys,values=[],[]
    for key,value in dictionnaire.items():
        keys.append(key)
        values.append(value)
    results=np.random.dirichlet(values)
    return {keys[i]:results[i] for i in range(len(dictionnaire))}



class PIM_Agent:

    def __init__(self,environment, gamma=0.95, beta=1, alpha=0.5, k=2):
        
        self.environment=environment
        self.alpha = alpha
        self.beta=beta
        self.gamma = gamma
        self.k=k
        
        self.R = defaultdict(lambda: defaultdict(lambda: 0.0)) #Récompenses
        self.tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0))) #Transitions
        
        self.nSA = defaultdict(lambda: defaultdict(lambda: 0.0)) #Compteur passage (état,action)
        self.nSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0))) #Compteur (état_1,action,état_2)

        
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0)) #Q-valeur état-action
        self.epis=defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.counter=self.nSA #Compteur (état,action)
        self.step_counter=0
        
        self.V= defaultdict(lambda: defaultdict(lambda: {}))
        self.policy=defaultdict(lambda: defaultdict(lambda: {}))
        
        self.planned=False
        self.k_max_epist={}
        self.k_max_rewards={}
        self.ajout_states()
            
    def learn(self,old_state,reward,new_state,action):
                    
                    self.nSA[old_state][action] +=1
                    self.R[old_state][action]=reward
                    self.nSAS[old_state][action][new_state] += 1
                    
                    
                    #Modifier les probabilités de transition selon le prior avec distribution de dirichlet
                    if self.nSA[old_state][action]==1:self.tSAS[old_state][action]=defaultdict(lambda:.0)                      
                    for next_state in self.nSAS[old_state][action]:
                        self.tSAS[old_state][action][next_state] = self.nSAS[old_state][action][next_state]/self.nSA[old_state][action]
                    
                    self.epis[old_state][action]=1
                    
                    #self.Q[old_state][action]=self.R[old_state][action]+self.gamma*np.sum([max(self.Q[next_state].values())*self.tSAS[old_state][action][next_state] for next_state in self.tSAS[old_state][action].keys()])                  
                    
                    if self.step_counter%30==0:
                        self.choose_between_goals()
    
    
    def get_k_max(self,dict_SA):
        d={(state,action):dict_SA[state][action] for state in self.states for action in self.environment.actions}
        l = list(d.items())
        random.shuffle(l)
        d = dict(l)
        max_index_k=sorted(d, key=d.get, reverse=True)[:self.k]
        return max_index_k
    
    def find_goals(self):
        max_reward_k=self.get_k_max(self.R)
        max_epis_k=self.get_k_max(self.epis)
        possible_goals_reward={max_reward:self.R[max_reward[0]][max_reward[1]] for max_reward in max_reward_k}
        possible_goals_epistemic={max_epis:self.alpha*self.epis[max_epis[0]][max_epis[1]] for max_epis in max_epis_k}
        possible_goals={**possible_goals_reward,**possible_goals_epistemic}
        print(possible_goals)
        return possible_goals
    
    def get_VI_goals(self):
        goals=self.find_goals()
        for goal,value in goals.items():
            R={(state[0],state[1],action):0 for state in self.states for action in self.environment.actions}
            R[goal]=value
            if self.V[goal[0],goal[1]][goal[2]]!={}:
                V={state:1 for state in self.states}
                policy={state:np.random.randint(5) for state in self.states}
            else : 
                V=self.V[goal[0],goal[1]][goal[2]]
                policy=self.policy[goal[0],goal[1]][goal[2]]
            self.V[goal[0],goal[1]][goal[2]],self.policy[goal[0],goal[1]][goal[2]]=self.value_iteration(V,policy,self.gamma,R)
        return goals
            
    def choose_between_goals(self):
        get_goals=self.get_VI_goals()
        current_loc=self.environment.current_location
        values_for_softmax={goal:self.V[goal[0],goal[1]][goal[2]][current_loc] for goal in get_goals}
        max_value=max(values_for_softmax.values())
        probas=np.zeros(len(get_goals))
        for goal in get_goals : 
            probas[goal]=self.values_for_softmax[goal]
        probas=np.exp(self.beta*(probas-max_value))
        chosen_policy = random.choices(get_goals,weights=probas,k=1)[0]
        self.active_policy=chosen_policy
        
    def choose_action(self): #argmax pour choisir l'action
        self.step_counter+=1
        return self.policy[self.active_policy[0],self.active_policy[1]][self.active_policy[2]][self.environment.current_location]
    
    
    def ajout_states(self):
        self.states=self.environment.states
        number_states=len(self.states)
        for state_1 in self.states:
            for action in self.environment.actions:
                for state_2 in self.states:
                    self.tSAS[state_1][action][state_2]=1/number_states
                self.Q[state_1][action]=(1+self.beta)/(1-self.gamma)
                self.epis[state_1][action]=0
                self.R[state_1][action]=0
        self.choose_between_goals()
                
    def value_iteration(self,V,policy,gamma,R,accuracy=0.01):
        delta=accuracy+1
        while delta > accuracy :
            delta=0
            for state,value in V.items():
                value_V=V[state]
                V[state]=np.max([np.sum([self.tSAS[state][action][new_state]*(R[state[0],state[1],action]+self.gamma*V[new_state]) for new_state in self.tSAS[state][action].keys()]) for action in self.environment.actions])
                delta=max(delta,np.abs(value_V-V[state]))
        for state,value in V.items():
            policy[state]=np.argmax([np.sum([self.tSAS[state][action][new_state]*(R[state[0],state[1],action]+self.gamma*V[new_state]) for new_state in self.tSAS[state][action].keys()]) for action in self.environment.actions])
        return V,policy
    
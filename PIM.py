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
                    
                    self.epis[old_state][action]=1/self.nSA[old_state][action]
                    
                    
                    if self.step_counter%self.environment.timescale==0:
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
        return possible_goals
    
    def get_VI_goals(self):
        goals=self.find_goals()
        for goal,value in goals.items():
            R={(state[0],state[1],action):0 for state in self.states for action in self.environment.actions}
            R[goal[0][0],goal[0][1],goal[1]]=value
            if self.V[goal[0]][goal[1]]=={}:
                V={state:1 for state in self.states}
                policy={state:np.random.randint(5) for state in self.states}
            else : 
                V=self.V[goal[0]][goal[1]]
                policy=self.policy[goal[0]][goal[1]]
            self.V[goal[0]][goal[1]],self.policy[goal[0]][goal[1]]=self.value_iteration(V,policy,self.gamma,R)
        return goals
            
    def choose_between_goals(self):
        get_goals=self.get_VI_goals()
        current_loc=self.environment.current_location
        values_for_softmax={goal:self.V[goal[0]][goal[1]][current_loc] for goal in get_goals}
        max_value=max(values_for_softmax.values())
        probas={goal: np.exp(self.beta*(values_for_softmax[goal]-max_value)) for goal in get_goals}
        somme_probas=sum(probas.values())
        probas={goal:value/somme_probas for goal,value in probas.items()}
        chosen_policy = random.choices(list(probas.keys()), weights=probas.values(), k=1)[0]
        self.active_policy=chosen_policy

        
    def choose_action(self): 
        self.step_counter+=1
        actions_possibles=self.policy[self.active_policy[0]][self.active_policy[1]][self.environment.current_location]
        action = random.choices(list(actions_possibles.keys()), weights=actions_possibles.values(), k=1)[0]
        """if (self.environment.current_location,action)==self.active_policy and self.R[self.environment.current_location][action]==0:
            self.choose_between_goals()"""
        return action
    
    
    def ajout_states(self):
        self.states=self.environment.states
        number_states=len(self.states)
        for state_1 in self.states:
            for action in self.environment.actions:
                for state_2 in self.states:
                    self.tSAS[state_1][action][state_2]=1/number_states
                self.epis[state_1][action]=10
                self.R[state_1][action]=0
                self.V[state_1][action]={}
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
            tab=np.array([np.sum([self.tSAS[state][action][new_state]*(R[state[0],state[1],action]+self.gamma*V[new_state]) for new_state in self.tSAS[state][action].keys()]) for action in self.environment.actions])
            best_actions=np.flatnonzero(tab == tab.max())
            policy[state]={action: 1/len(best_actions) for action in best_actions}
        return V,policy
    
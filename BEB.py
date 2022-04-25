import numpy as np
from collections import defaultdict


def count_to_dirichlet(dictionnaire):
    keys,values=[],[]
    for key,value in dictionnaire.items():
        keys.append(key)
        values.append(value)
    results=np.random.dirichlet(values)
    return {keys[i]:results[i] for i in range(len(dictionnaire))}



class BEB_Agent:

    def __init__(self,environment, gamma=0.95, beta=1,known_states=False,coeff_prior=0.5,optimistic=0.5,informative=False):
        
        self.environment=environment
        self.gamma = gamma
        self.beta=beta
        
        self.R = defaultdict(lambda: defaultdict(lambda: 0.0)) #Récompenses
        self.Rsum = defaultdict(lambda: defaultdict(lambda: 0.0)) #Somme récompenses accumulées 
        self.tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0))) #Transitions
        
        self.nSA = defaultdict(lambda: defaultdict(lambda: 0.0)) #Compteur passage (état,action)
        self.nSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0))) #Compteur (état_1,action,état_2)

        
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0)) #Q-valeur état-action
        self.bonus=defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.counter=self.nSA #Compteur (état,action)
        self.step_counter=0
        
        self.known_states=known_states
        self.prior=defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.coeff_prior=coeff_prior
        self.optimistic=optimistic
        self.informative=informative
        if self.known_states: self.ajout_states()
            
    def learn(self,old_state,reward,new_state,action):
                    
                    self.uncountered_state(new_state) #Si l'état est nouveau, création des q-valeurs pour les actions possibles
                    
                    self.nSA[old_state][action] +=1
                    self.Rsum[old_state][action] += reward
                    self.nSAS[old_state][action][new_state] += 1
                    self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]
                    self.prior[old_state][action][new_state]+=1
                    
                    #Modifier les probabilités de transition selon le prior avec distribution de dirichlet
                    self.tSAS[old_state][action]=count_to_dirichlet(self.prior[old_state][action])
                    
                    #Ajout du bonus qui dépend du nombre de passages
                    self.bonus[old_state][action]=self.beta/(1+self.nSA[old_state][action])
                    
                    self.Q[old_state][action]=self.R[old_state][action]+self.bonus[old_state][action]+self.gamma*np.sum([max(self.Q[next_state].values())*self.tSAS[old_state][action][next_state] for next_state in self.tSAS[old_state][action].keys()])                  

    def choose_action(self): #argmax pour choisir l'action
        self.step_counter+=1
        state=self.environment.current_location
        self.uncountered_state(state)
        q_values = self.Q[state]
        maxValue = max(q_values.values())
        action = np.random.choice([k for k, v in q_values.items() if v == maxValue])
        return action
    
    
    def uncountered_state(self,state): #création des q-valeurs pour les actions possibles pour les nouveaux états
        known_states=self.prior.keys()
        if state not in known_states:
            for move in self.environment.actions:
                self.bonus[state][move]=self.beta
                self.Q[state][move]=self.beta*self.optimistic
    
    def ajout_states(self):
        self.states=self.environment.states
        for state_1 in self.states:
            for action in self.environment.actions:
                if self.informative : 
                    prior_transitions=self.environment.transitions[action][state_1].copy()
                    prior_coeff_transitions={k:max(self.coeff_prior*v,1e-5) for k,v in prior_transitions.items()}
                    self.prior[state_1][action]=prior_coeff_transitions
                    """for state_2 in self.environment.transitions[action][state_1].keys():
                        self.prior[state_1][action][state_2]=self.coeff_prior"""                        
                else : 
                    for state_2 in self.states : self.prior[state_1][action][state_2]=self.prior[state_1][action][state_2]=self.coeff_prior
                self.bonus[state_1][action]=self.beta
                self.Q[state_1][action]=self.beta*self.optimistic
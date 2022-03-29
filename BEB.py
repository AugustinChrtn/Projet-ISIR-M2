import numpy as np
from collections import defaultdict

class BEB_Agent:

    def __init__(self,environment, gamma=0.95, beta=1):
        
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
        
    def learn(self,old_state,reward,new_state,action):
                    
                    self.uncountered_state(new_state) #Si l'état est nouveau, création des q-valeurs pour les actions possibles
                    
                    self.nSA[old_state][action] +=1
                    self.Rsum[old_state][action] += reward
                    self.nSAS[old_state][action][new_state] += 1
                    self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]
                    
                    #Modifier les probabilités de transition selon le compteur de passages avec distribution de dirichlet
                    keys=[]
                    values=[]
                    for next_state, next_state_count in self.nSAS[old_state][action].items():
                        keys.append(next_state)
                        values.append(next_state_count)
                    values=np.random.dirichlet(values)
                    for i in range(len(keys)):
                        self.tSAS[old_state][action][keys[i]]=values[i]
                        
                    #Sans dirichlet! : Moins de variabilité et marche un peu moins bien.
                    """for next_state in self.nSAS[old_state][action].keys()::
                            self.tSAS[old_state][action][next_state] = self.nSAS[old_state][action][next_state]/self.nSA[old_state][action]"""
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
        known_states=self.nSA.keys()
        if state not in known_states:
            for move in self.environment.actions:
                self.bonus[state][move]=self.beta
                self.Q[state][move]=self.beta
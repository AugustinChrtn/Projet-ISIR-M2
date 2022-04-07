import numpy as np
from collections import defaultdict


class QMB_Agent:

    def __init__(self,environment, gamma=0.95,epsilon = 0.1,H_update=3,entropy_factor=10,gamma_epis=0.5,epis_factor=10):
        
        self.environment=environment
        self.gamma = gamma
        self.epsilon = epsilon
        self.gamma_epis=gamma_epis
        
        self.R = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.nSA = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.nSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        self.Rsum = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.counter=self.nSA
        self.step_counter=0
        
        self.H=defaultdict(lambda: defaultdict(lambda: 0.0))
        self.tSAS_old=defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        self.KL=defaultdict(lambda: defaultdict(lambda: 0.0))
        self.H_update=H_update
        
        self.epis=defaultdict(lambda: defaultdict(lambda: 0.0))
        self.epis_factor=epis_factor
        self.entropy_factor=entropy_factor
        
    def learn(self,old_state,reward,new_state,action):
                    
                    self.uncountered_state(new_state)
                    if self.nSA[old_state][action]==0:self.tSAS_old[old_state][action][new_state]=1
                    
                    self.nSA[old_state][action] +=1
                    self.Rsum[old_state][action] += reward
                    self.nSAS[old_state][action][new_state] += 1
                    self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]  

                    self.tSAS[old_state][action]=defaultdict(lambda:0.0)   
                    for next_state in self.nSAS[old_state][action].keys():
                        self.tSAS[old_state][action][next_state] = self.nSAS[old_state][action][next_state]/self.nSA[old_state][action]

                    self.Q[old_state][action]=self.R[old_state][action]+self.gamma_epis*np.sum([max(self.Q[next_state].values())*self.tSAS[old_state][action][next_state] for next_state in self.tSAS[old_state][action].keys()])
                    
                    self.H[old_state][action]=self.entropy(self.tSAS[old_state][action])
                    if self.counter[old_state][action]%self.H_update==1:
                        self.KL[old_state][action]=self.KL_div(self.tSAS_old[old_state][action],self.tSAS[old_state][action])
                        self.tSAS_old[old_state][action] = self.tSAS[old_state][action]
                    
                    reward_epis=self.entropy_factor*self.H[old_state][action]+self.KL[old_state][action]               
                    self.epis[old_state][action] = reward_epis+ self.gamma*np.sum([max(self.epis[next_state].values())*self.tSAS[old_state][action][next_state] for next_state in self.tSAS[old_state][action].keys()])
                    
    def choose_action(self):
        self.step_counter+=1
        state=self.environment.current_location
        
        self.uncountered_state(state)
        
        if np.random.random() > (1-self.epsilon) :
            action = np.random.choice(self.environment.actions)
        else:
                q_values = self.Q[state]
                epis_values=self.epis[state]
                total_dict={}
                for key in q_values.keys():
                    total_dict[key]=self.epis_factor*(epis_values[key])+q_values[key]
                maxValue = max(total_dict.values())
                action = np.random.choice([k for k, v in total_dict.items() if v == maxValue])
        return action
    
    def uncountered_state(self,state):
        known_states=self.nSA.keys()
        if state not in known_states:
            for action in self.environment.actions:
                self.R[state][action]=0
                self.Q[state][action]=0
                self.H[state][action]=0
                self.KL[state][action]=0 
                self.epis[state][action]=0
    
    def entropy(self,transitions):
        value_entropy=0
        for value in transitions.values():
            if value>0:
                value_entropy+=-value*np.log2(value)
        return value_entropy
    
    def KL_div(self,transi_1,transi_2):
        value_KL=0
        for key,value in transi_1.items():
            value_KL+=value*np.log2(value/transi_2[key])
        return value_KL
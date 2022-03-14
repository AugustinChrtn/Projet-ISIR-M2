import numpy as np
from collections import defaultdict

class BEB_Agent:

    def __init__(self,environment, gamma=0.95, beta=1):
        
        self.environment=environment
        self.gamma = gamma
        self.beta=beta
        
        self.R = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.nSA = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.nSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        self.Rsum = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.qSA = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.counter=self.nSA
        
    def learn(self,old_state,reward,new_state,action):
                    
                    self.uncountered_state(new_state)
                    
                    self.nSA[old_state][action] +=1
                    self.Rsum[old_state][action] += reward
                    self.nSAS[old_state][action][new_state] += 1
                    self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]
                    keys=[]
                    values=[]
                    for next_state, next_state_count in self.nSAS[old_state][action].items():
                        keys.append(next_state)
                        values.append(next_state_count)
                    values=np.random.dirichlet(values)
                    for i in range(len(keys)):
                        self.tSAS[old_state][action][keys[i]]=values[i]
                        
                    self.qSA[old_state][action]=self.R[old_state][action]+self.beta/(1+self.nSA[old_state][action])+self.gamma*np.sum([max(self.qSA[next_state].values())*self.tSAS[old_state][action][next_state] for next_state in self.tSAS[old_state][action].keys()])
                                        

    def choose_action(self):
        state=self.environment.current_location
        self.uncountered_state(state)
        q_values = self.qSA[state]
        maxValue = max(q_values.values())
        action = np.random.choice([k for k, v in q_values.items() if v == maxValue])
        return action
    
    def uncountered_state(self,state):
        if state not in self.nSA.keys():
            for move in self.environment.actions:
                self.qSA[state][move]=0
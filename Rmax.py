import numpy as np
from collections import defaultdict


class Rmax_Agent:

    def __init__(self,environment, gamma=0.95, max_visits_per_state=10, epsilon = 0.1,Rmax=200):
        
        self.Rmax=Rmax
        
        self.environment=environment
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_visits_per_state = max_visits_per_state    
        
        self.R = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.Rsum = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.nSA = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.nSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.counter=self.nSA
        self.step_counter=0
        
    def learn(self,old_state,reward,new_state,action):
                    
                    self.uncountered_state(new_state)
                    
                    self.nSA[old_state][action] +=1
                    self.Rsum[old_state][action] += reward
                    self.nSAS[old_state][action][new_state] += 1

                    if self.nSA[old_state][action] >= self.max_visits_per_state :
                        self.tSAS[old_state][action]=defaultdict(lambda:.0)
                        self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]                         
                        for next_state in self.nSAS[old_state][action].keys():
                            self.tSAS[old_state][action][next_state] = self.nSAS[old_state][action][next_state]/self.nSA[old_state][action]

                    self.Q[old_state][action]=self.R[old_state][action]+self.gamma*np.sum([max(self.Q[next_state].values())*self.tSAS[old_state][action][next_state] for next_state in self.tSAS[old_state][action].keys()])

    def choose_action(self):
        self.step_counter+=1
        state=self.environment.current_location
        self.uncountered_state(state)
        if np.random.random() > (1-self.epsilon) :
            action = np.random.choice(self.environment.actions)
        else:
                q_values = self.Q[state]
                maxValue = max(q_values.values())
                action = np.random.choice([k for k, v in q_values.items() if v == maxValue])
        return action
    
    def uncountered_state(self,state):
        if state not in self.R.keys():
            for action in self.environment.actions : 
                self.R[state][action]=self.Rmax
                self.tSAS[state][action][state]=1
                self.Q[state][action]=self.Rmax
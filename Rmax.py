import numpy as np
from collections import defaultdict


class Rmax_Agent:

    def __init__(self,environment, gamma=0.95, m=5,Rmax=200,known_states=True,optimistic=1):
        
        self.Rmax=Rmax
        
        self.environment=environment
        self.gamma = gamma
        self.m = m   
        
        self.R = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.Rsum = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.nSA = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.nSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.counter=self.nSA
        self.step_counter=0
        
        self.prior = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        self.optimistic=optimistic
        self.max_visits=defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))

        if known_states : self.ajout_states()

        
    def learn(self,old_state,reward,new_state,action):
                    
                    self.uncountered_state(new_state)
                    
                    self.nSA[old_state][action] +=1
                    self.Rsum[old_state][action] += reward
                    self.nSAS[old_state][action][new_state] += 1
                    self.prior[old_state][action][new_state]+=1
                    

                    if self.nSA[old_state][action] >= self.max_visits[old_state][action] :
                        self.tSAS[old_state][action]=defaultdict(lambda:.0)
                        self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]                         
                        for next_state in self.nSAS[old_state][action].keys():
                            self.tSAS[old_state][action][next_state] = self.prior[old_state][action][next_state]/sum(self.prior[old_state][action].values())

                    self.Q[old_state][action]=self.R[old_state][action]+self.gamma*np.sum([max(self.Q[next_state].values())*self.tSAS[old_state][action][next_state] for next_state in self.tSAS[old_state][action].keys()])

    def choose_action(self):
        self.step_counter+=1
        state=self.environment.current_location
        self.uncountered_state(state)
        q_values = self.Q[state]
        maxValue = max(q_values.values())
        action = np.random.choice([k for k, v in q_values.items() if v == maxValue])
        return action
    
    def uncountered_state(self,state):
        if state not in self.R.keys():
            for action in self.environment.actions : 
                self.R[state][action]=self.Rmax
                self.tSAS[state][action][state]=1
                self.Q[state][action]=self.Rmax*self.optimistic
                self.max_visits[state][action]=self.m
    
    def ajout_states(self):
        self.states=self.environment.states
        for state_1 in self.states:
            for action in self.environment.actions:
                self.tSAS[state_1][action][state_1]=1
                self.R[state_1][action]=self.Rmax
                self.Q[state_1][action]=self.Rmax*self.optimistic
                self.max_visits[state_1][action]=self.environment.entropy[state_1,action]*self.m
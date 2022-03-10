import numpy as np
from collections import defaultdict
UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4

def create_matrix(width,height,liste):
    return [[liste for i in range(width)] for j in range(height)]
    

class Rmax_Agent:

    def __init__(self,environment, gamma=0.95, max_visits_per_state=10, epsilon = 0.1):
        
        self.Rmax=200
        
        self.environment=environment
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_visits_per_state = max_visits_per_state    
        self.counter=np.array(create_matrix(environment.width, environment.height,[0,0,0,0,0])) 
        
        self.R = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.nSA = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.nSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        self.Rsum = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.qSA = defaultdict(lambda: defaultdict(lambda: 0.0))
        
                    
    def learn(self,old_state,reward,new_state,action):
                    
                    if self.nSA[old_state][action] < self.max_visits_per_state :

                        self.nSA[old_state][action] +=1
                        self.Rsum[old_state][action] += reward
                        self.nSAS[old_state][action][new_state] += 1

                    else :

                        self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]                         
                        for next_state in self.nSAS.keys():
                            self.tSAS[old_state][action][next_state] = self.nSAS[old_state][action][next_state]/self.nSA[old_state][action]
                    
                    self.qSA[old_state][action]=self.R[old_state][action]+self.gamma*np.sum([max(self.qSA[next_state].values())*self.tSAS[old_state][action][next_state] for next_state in self.tSAS[old_state][action].keys()])
                                        

    def choose_action(self):
        state=self.environment.current_location
        if state not in self.R.keys():
            for action in self.environment.actions : 
                self.R[state][action]=self.Rmax
                self.tSAS[state][action][state]=1
                self.qSA[state][action]=0
        if np.random.random() > (1-self.epsilon) :
            action = np.random.choice(self.environment.actions)
        else:
                q_values = self.qSA[state]
                maxValue = max(q_values.values())
                action = np.random.choice([k for k, v in q_values.items() if v == maxValue])
        self.counter[self.environment.current_location][action]+=1
        return action
import numpy as np
from collections import defaultdict
UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4

def create_matrix(width,height,liste):
    return [[liste for i in range(width)] for j in range(height)]

def convert_default_dict(dic):
    if isinstance(dic,defaultdict):
        return dict((key,convert_default_dict(value)) for key,value in dic.items())
    else : return dic
    
"""
RMAX’s sketch
Initialize all couter n(s, a) = 0, n(s, a, s′) = 0.
Initialize ˆT (s′|s, a) = Is=s′ , ˆR(s, a) = Rmax
while (1) do
Compute policy πt using MDP model of ( ˆT , ˆR).
Choose a = πt(s), observe s′, r.
n(s, a) = n(s, a) + 1
r(s, a) = r(s, a) + r
n(s, a, s′) = n(s, a, s′) + 1
if n(s, a) = m then
Update ˆT (·|s, a) = n(s, a, ·)/n(s, a), and ˆR(s, a) = r(s, a)/n(s, a)."""


class Rmax_Agent:

    def __init__(self,environment, gamma, max_visits_per_state, epsilon = 0.2):
        
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
        
                    
    def learn(self,old_state,reward,new_state,action):
                    
                    if self.nSA[old_state][action] < self.max_visits_per_state :

                        self.nSA[old_state][action] +=1
                        self.Rsum[old_state][action] += reward
                        self.nSAS[old_state][action][new_state] += 1

                    else :

                        self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]                         
                        for next_state in self.nSAS.keys():
                            self.tSAS[old_state][action][next_state] = self.nSAS[old_state][action][next_state]/self.nSA[old_state][action]
                
                    
                    
    """Compute policy πt using MDP model of ( ˆT , ˆR).
        Choose a = πt(s), observe s′, r."""

    def choose_action(self):
        state=self.environment.current_location
        if state not in self.R.keys():
            for action in self.environment.actions : 
                self.R[state][action]=self.Rmax
                self.nSA[state][action] =0
                self.Rsum[state][action]=0
                self.tSAS[state][action][state]=1
        if np.random.random() > (1-self.epsilon) :
            action = np.random.choice(self.environment.actions)
        else:
                expected_rewards = [np.sum(R[environment.current_location[0]][environment.current_location[1]][action])]
                action=np.random.choice(np.flatnonzero(q_values == q_values.max()))
        self.counter[self.environment.current_location][action]+=1
        return action
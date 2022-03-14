import numpy as np
import random
LEFT,RIGHT=0,1



class Two_step:
    
    def __init__(self):
        
        self.actions=[LEFT,RIGHT]
        self.first_location=0
        
        self.nS=5
        self.nA=2
        
        self.P=np.zeros((self.nS,self.nA,self.nS))
        
        self.P[0, LEFT, 1] = 1.0
        self.P[1, LEFT, 3] = 0.9
        self.P[1, LEFT, 4] = 0.1
        self.P[2, LEFT, 3] = 0.5
        self.P[2, LEFT, 4] = 0.5
        self.P[3, LEFT, 0] = 1.0
        self.P[4, LEFT, 0] = 1.0

        self.P[0, RIGHT, 2] = 1.0
        self.P[1, RIGHT, 4] = 0.9
        self.P[1, RIGHT, 3] = 0.1
        self.P[2, RIGHT, 4] = 0.5
        self.P[2, RIGHT, 3] = 0.5
        self.P[3, RIGHT, 0] = 1.0
        self.P[4, RIGHT, 0] = 1.0
        
        self.R=np.zeros((self.nS,self.nA))
        self.R[4,:]=1
        self.final_states=[(4,RIGHT),(4,LEFT),(3,RIGHT),(3,LEFT)]
        self.current_location = self.first_location
        
    def make_step(self, action):
        state = self.current_location       
        reward = self.R[state][action]
        self.current_location=random.choices([0,1,2,3,4],weights=self.P[state,action,:],k=1)[0]
        return reward, (state,action) in self.final_states
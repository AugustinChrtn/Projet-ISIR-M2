import numpy as np
from collections import defaultdict


class Kalman_agent_sum(): 
    
    def __init__(self, environment, gamma=1, variance_ob=1,variance_tr=40,curiosity_factor=2):
        self.environment = environment
        self.K_mean = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.K_var = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.counter= defaultdict(lambda: defaultdict(lambda: 0.0))
        self.gamma = gamma
        self.variance_ob=variance_ob
        self.variance_tr=variance_tr
        self.curiosity_factor=curiosity_factor

    def choose_action(self):
        state=self.environment.current_location
        actions=self.environment.actions
        self.uncountered_state(state)
        get_mean=self.K_mean[state]
        get_variance=self.K_var[state]
        probas=np.zeros(len(actions))
        for move in actions:
            probas[move]=np.random.normal(get_mean[move],np.sqrt(get_variance[move]))+(self.curiosity_factor*get_variance[move])
        action = np.argmax(probas)
        self.counter[state][action]+=1
        return action
    
    def learn(self, old_state, reward, new_state, action):
        self.uncountered_state(new_state)
        max_mean_in_new_state = max(self.K_mean[new_state].values())
        current_mean = self.K_mean[old_state][action]
        current_variance = self.K_var[old_state][action]
        self.K_mean[old_state][action] = ((current_variance+self.variance_tr)*(reward +self.gamma * max_mean_in_new_state)+(self.variance_ob*current_mean))/ (current_variance+self.variance_tr+self.variance_ob)
        self.K_var[old_state][action]=((current_variance+self.variance_tr)*self.variance_ob)/(current_variance+self.variance_ob+self.variance_tr)
        for move in self.environment.actions:
            if move != action : 
                self.K_var[old_state][move]+=self.variance_tr
    
    def uncountered_state(self,state):
        if state not in self.K_mean.keys():
            for move in self.environment.actions:
                self.K_mean[state][move]=0
                self.K_var[state][move]=1

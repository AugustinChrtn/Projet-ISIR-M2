import numpy as np
import random
from collections import defaultdict


class Q_Agent():
    def __init__(self, environment, alpha=0.95, beta=0.0002, gamma=1, epsilon=0.1, exploration='softmax'):
        
        self.environment = environment
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.counter=defaultdict(lambda: defaultdict(lambda: 0.0))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.exploration=exploration
        self.beta=beta
        
    def choose_action(self):       
        
        state=self.environment.current_location
        actions=self.environment.actions
        self.uncountered_state(state)     
        
        if self.exploration=='e-greedy':
            if np.random.rand() < self.epsilon:
                action = np.random.choice(actions)
            else:
                q_values = self.Q[state]
                maxValue = max(q_values.values())
                action = np.random.choice([k for k, v in q_values.items() if v == maxValue])
        
        elif self.exploration=='softmax':
            max_value=max(self.Q[state].values())
            probas=np.zeros(len(actions))
            for action in actions : 
                probas[action]=self.Q[state][action]
            probas=np.exp(self.beta*(probas-max_value))
            action = random.choices(actions,weights=probas,k=1)[0]
        
        self.counter[state][action]+=1
        return action
        
    def learn(self, old_state, reward, new_state, action):       
        self.uncountered_state(new_state)
        max_q_value_in_new_state = max(self.Q[new_state].values())
        current_q_value = self.Q[old_state][action]
        self.Q[old_state][action] = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * max_q_value_in_new_state)
    
    def uncountered_state(self,state):
        if state not in self.Q.keys():
            for move in self.environment.actions:
                self.Q[state][move]=0
import numpy as np
import random

def create_matrix(width,height,liste):
    return [[liste for i in range(width)] for j in range(height)]

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4



class Q_Agent():
    def __init__(self, environment, alpha=0.95, beta=0.0002, gamma=1, epsilon=0.1, exploration='softmax'):
        self.environment = environment
        self.q_table =np.array(create_matrix(environment.width, environment.height,[0,0,0,0,0])) 
        self.counter=np.array(create_matrix(environment.width, environment.height,[0,0,0,0,0])) 
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.exploration=exploration
        self.beta=beta
        
    def choose_action(self):
        if self.exploration=='e-greedy':
            if np.random.rand() < self.epsilon:
                action = np.random.choice([UP,DOWN,LEFT,RIGHT,STAY])
            else:
                q_values = self.q_table[self.environment.current_location]
                action=np.random.choice(np.flatnonzero(q_values == q_values.max()))
        elif self.exploration=='softmax':
            q_values = self.q_table[self.environment.current_location]
            max_value=q_values.max()
            probas=np.exp(self.beta*(q_values.copy()-max_value))
            action = random.choices([UP,DOWN,LEFT,RIGHT,STAY],weights=probas,k=1)[0]
        self.counter[self.environment.current_location][action]+=1
        return action
        
    def learn(self, old_state, reward, new_state, action):
        q_values_of_state = self.q_table[new_state]
        max_q_value_in_new_state = np.max(q_values_of_state)
        current_q_value = self.q_table[old_state][action]        
        self.q_table[old_state][action] = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * max_q_value_in_new_state)

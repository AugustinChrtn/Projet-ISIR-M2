import numpy as np
import random


def create_matrix(width,height,liste):
    return [[liste for i in range(width)] for j in range(height)]

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4

def choice_dictionary(dictionary):
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    return random.choices(keys, weights=values)[0]

def entropy(transitions):
    value_entropy=0
    for value in transitions.values():
        if value>0:
            value_entropy+=-value*np.log2(value)
    return value_entropy

class Lopes_State():
    def __init__(self,transitions):
        self.height = 5
        self.width = 5
        self.grid = np.zeros((self.height,self.width))        
        self.final_states={}
        self.values= np.zeros((self.height,self.width))
        self.current_location = (0,0)     
        self.first_location=(0,0)
        
        self.actions = [UP, DOWN, LEFT, RIGHT,STAY]
        malus=-0.1
        bonus=1
        self.rewards={(1,1):malus, (1,2):malus,(1,3):malus,(2,2):malus,(2,4):bonus}

        
        for state, reward in self.rewards.items():
            self.values[state]=reward

        self.max_exploration=125
        self.UP=transitions[UP]
        self.DOWN=transitions[DOWN]
        self.LEFT=transitions[LEFT]
        self.RIGHT=transitions[RIGHT]
        self.STAY=transitions[STAY]
                
        self.states=[(i,j) for i in range(self.height) for j in range(self.width)]
        self.transitions=transitions
        self.uncertain_states=[(0,1),(0,3),(2,1),(2,3)]
        self.entropy={(state,action) : entropy(transitions[action][state]) for state in self.states for action in self.actions }
        
    def make_step(self, action):
        last_location = self.current_location       
        if action == UP:
            self.current_location = choice_dictionary(self.UP[last_location[0]][last_location[1]])        
        elif action == DOWN:
            self.current_location = choice_dictionary(self.DOWN[last_location[0]][last_location[1]])   
        elif action == LEFT:
            self.current_location = choice_dictionary(self.LEFT[last_location[0]][last_location[1]]) 
        elif action == RIGHT:
            self.current_location = choice_dictionary(self.RIGHT[last_location[0]][last_location[1]]) 
        elif action == STAY:
            self.current_location = choice_dictionary(self.STAY[last_location[0]][last_location[1]])
        return self.values[self.current_location], (last_location[0],last_location[1],action) in self.final_states.keys()
        
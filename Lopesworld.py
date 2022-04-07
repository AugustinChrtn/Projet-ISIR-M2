import numpy as np
import random


def create_matrix(width,height,liste):
    return [[liste for i in range(width)] for j in range(height)]

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4

def choice_dictionary(dictionary):
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    return random.choices(keys, weights=values)[0]


class Lopes_State():
    def __init__(self,transitions):
        self.height = 5
        self.width = 5
        self.grid = np.zeros((self.height,self.width))        
        self.final_states={(2,4,STAY):1}
        self.values= np.array(create_matrix(self.width, self.height,[0,0,0,0,0]))
        self.current_location = (0,0)     
        self.first_location=(0,0)
        
        self.actions = [UP, DOWN, LEFT, RIGHT,STAY]
        
        self.rewards={(1,1,UP):-0.1,(1,1,LEFT):-0.1,(1,1,RIGHT):-0.1,(1,1,DOWN):-0.1,(1,1,STAY):-0.1,(2,4,STAY):1,
                      (1,2,UP):-0.1,(1,2,LEFT):-0.1,(1,2,RIGHT):-0.1,(1,2,DOWN):-0.1,(1,2,STAY):-0.1,
                      (1,3,UP):-0.1,(1,3,LEFT):-0.1,(1,3,RIGHT):-0.1,(1,3,DOWN):-0.1,(1,3,STAY):-0.1,
                      (2,2,UP):-0.1,(2,2,LEFT):-0.1,(2,2,RIGHT):-0.1,(2,2,DOWN):-0.1,(2,2,STAY):-0.1}
        
        for transition, reward in self.rewards.items():
            self.values[transition[0]][transition[1]][transition[2]]=reward
        
        
        self.UP=transitions[UP]
        self.DOWN=transitions[DOWN]
        self.LEFT=transitions[RIGHT]
        self.RIGHT=transitions[LEFT]
        self.STAY=transitions[STAY]
                
            
    def make_step(self, action):
        last_location = self.current_location       
        reward = self.values[last_location[0]][last_location[1]][action]
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
        return reward, (last_location[0],last_location[1],action) in self.final_states.keys()
        
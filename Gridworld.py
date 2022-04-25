import numpy as np
import random

def create_matrix(width,height,liste):
    return [[liste for i in range(width)] for j in range(height)]

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4

def choice_dictionary(dictionary):
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    return random.choices(keys, weights=values)[0]

class State:
    def __init__(self):
        self.height = 5
        self.width = 5
        self.grid = np.zeros((self.height,self.width))        
        self.final_states={(4,4,STAY):1}
        self.values= np.array(create_matrix(self.width, self.height,[0,0,0,0,0]))  
        self.current_location = (0,0)     
        self.first_location=(0,0)
        for transition, reward in self.final_states.items():
            self.values[transition[0]][transition[1]][transition[2]]=reward
        self.actions = [UP, DOWN, LEFT, RIGHT,STAY]
        
        
        self.states=[(i,j) for i in range(self.height) for j in range(self.width)]
        self.UP=[list({} for i in range(self.width))for i in range(self.height)]
        self.DOWN=[list({} for i in range(self.width))for i in range(self.height)]
        self.LEFT=[list({} for i in range(self.width))for i in range(self.height)]
        self.RIGHT=[list({} for i in range(self.width))for i in range(self.height)]
        self.STAY=[list({} for i in range(self.width))for i in range(self.height)]
        
        for row in range(self.height):
            for col in range(self.width):
                if row==0 or (row-1,col): self.UP[row][col][row,col]=1
                else : self.UP[row][col][row-1,col]=1
                if row==self.height-1 : self.DOWN[row][col][row,col]=1
                else : self.DOWN[row][col][row+1,col]=1
                if col==0 :self.LEFT[row][col][row,col]=1
                else : self.LEFT[row][col][row,col-1]=1
                if col==self.width-1: self.RIGHT[row][col][row,col]=1
                else : self.RIGHT[row][col][row,col+1]=1
                self.STAY[row][col][row,col]=1
        
        self.transitions=np.array([self.UP,self.DOWN,self.LEFT,self.RIGHT,self.STAY])
        self.max_exploration=len(self.actions)*self.height*self.width
              
    def make_step(self, action):
        last_location = self.current_location       
        reward = self.values[last_location[0]][last_location[1]][action]
        if action == UP: self.current_location = choice_dictionary(self.UP[last_location[0]][last_location[1]])        
        elif action == DOWN: self.current_location = choice_dictionary(self.DOWN[last_location[0]][last_location[1]])   
        elif action == LEFT: self.current_location = choice_dictionary(self.LEFT[last_location[0]][last_location[1]]) 
        elif action == RIGHT: self.current_location = choice_dictionary(self.RIGHT[last_location[0]][last_location[1]]) 
        elif action == STAY: self.current_location = choice_dictionary(self.STAY[last_location[0]][last_location[1]])
        return reward, (last_location[0],last_location[1],action) in self.final_states.keys()

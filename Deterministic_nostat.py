import numpy as np
import random


def create_matrix(width,height,liste):
    return [[liste for i in range(width)] for j in range(height)]

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4

def choice_dictionary(dictionary):
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    return random.choices(keys, weights=values)[0]


class Deterministic_no_stat:
    
    def __init__(self,world,world2):
        
        
        self.world=world
        self.world2=world2
        
        
        initial_state=[]
        final_state=[]
        wall_state=[]
        for row in range(len(world)):
            for col in range(len(world[0])):
                if world[row,col]==-1:wall_state.append((row,col))
                if world[row,col]==-2:initial_state.append((row,col))
                if world[row,col]>0:final_state.append((row,col))
        self.height = len(world)
        self.width = len(world[0])
        self.grid = world       
        self.final_states={}
        for state in final_state:self.final_states[state[0],state[1],STAY]=world[state]
        self.values= np.array(create_matrix(self.width, self.height,[-0.01,-0.01,-0.01,-0.01,-0.01]))  
        self.current_location = initial_state[0]    
        self.first_location=initial_state[0]
        for transition, reward in self.final_states.items():
            self.values[transition[0],transition[1],STAY]=reward
        self.walls=wall_state

        self.actions = [UP, DOWN, LEFT, RIGHT,STAY]
        self.UP=[list({} for i in range(self.width))for i in range(self.height)]
        self.DOWN=[list({} for i in range(self.width))for i in range(self.height)]
        self.LEFT=[list({} for i in range(self.width))for i in range(self.height)]
        self.RIGHT=[list({} for i in range(self.width))for i in range(self.height)]
        self.STAY=[list({} for i in range(self.width))for i in range(self.height)]
        
        for row in range(self.height):
            for col in range(self.width):
                if row==0 or (row-1,col) in self.walls: self.UP[row][col][row,col]=1
                else : self.UP[row][col][row-1,col]=1
                if row==self.height-1 or (row+1,col) in self.walls: self.DOWN[row][col][row,col]=1
                else : self.DOWN[row][col][row+1,col]=1
                if col==0 or (row,col-1) in self.walls:self.LEFT[row][col][row,col]=1
                else : self.LEFT[row][col][row,col-1]=1
                if col==self.width-1 or (row,col+1) in self.walls: self.RIGHT[row][col][row,col]=1
                else : self.RIGHT[row][col][row,col+1]=1
                self.STAY[row][col][row,col]=1
        
        self.states=[]
        for i in range(self.height):
            for j in range(self.width):
                if (i,j) not in self.walls:
                    self.states.append((i,j))
        self.transitions=np.array([self.UP,self.DOWN,self.LEFT,self.RIGHT,self.STAY])
        self.max_exploration=len(self.actions)*(self.height*self.width-len(self.walls))
        self.number_steps=0
        self.changed=False
        
        
    def make_step(self, action):
        self.number_steps+=1
        if self.number_steps>5000 and not self.changed and self.current_location==self.first_location :
            number_steps=self.number_steps
            self.__init__(self.world2,self.world)
            self.changed=True
            self.number_steps=number_steps
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
        
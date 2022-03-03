import numpy as np

def create_matrix(width,height,liste):
    return [[liste for i in range(width)] for j in range(height)]

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4

class State:
    def __init__(self):
        self.height = 5
        self.width = 5
        self.grid = np.zeros((self.height,self.width))        
        self.final_states={(3,3,UP):10}
        self.values= np.array(create_matrix(self.width, self.height,[-1,-1,-1,-1,-1]))  
        self.current_location = (0,0)     
        self.first_location=(0,0)
        for transition, reward in self.final_states.items():
            self.values[transition[0]][transition[1]][transition[2]]=reward
        self.actions = [UP, DOWN, LEFT, RIGHT,STAY]
              
    def make_step(self, action):
        last_location = self.current_location       
        reward = self.values[last_location[0]][last_location[1]][action]
        if action == UP:
            if last_location[0] != 0:
                self.current_location = ( last_location[0] - 1, last_location[1])        
        elif action == DOWN:
            if last_location[0] != self.height - 1:
                self.current_location = ( last_location[0] + 1, last_location[1])
        elif action == LEFT:
            if last_location[1] != 0:
                self.current_location = ( last_location[0], last_location[1] - 1)
        elif action == RIGHT:
            if last_location[1] != self.width - 1:
                self.current_location = ( last_location[0], last_location[1] + 1)        
        return reward, (last_location[0],last_location[1],action) in self.final_states.keys()

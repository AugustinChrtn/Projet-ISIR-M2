import numpy as np

def create_matrix(width,height,liste):
    return [[liste for i in range(width)] for j in range(height)]

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4
class ComplexState:
    def __init__(self):
        self.height = 20
        self.width = 20
        self.grid=np.zeros((self.height,self.width))
        self.obstacles=[(0,9),(0,10),(1,9),(1,10),(2,9),(2,10),(3,0),(3,1),(3,2),
                        (3,3),(3,9),(3,10),(4,0),(4,1),(4,2),(4,3),(4,4),(4,9),(4,10),(4,15),
                        (4,16),(4,17),(4,18),(4,19),(5,15),(5,16),(5,17),(5,18),(5,19),(7,5),(7,6),(8,5),(8,6),(10,9),
                        (10,10),(10,13),(10,14),(10,15),(10,16),(10,17),(10,18),(10,19),(11,9),(11,10),(11,13),(11,14),
                        (11,15),(11,16),(11,17),(11,18),(11,19),(12,0),(12,1),(12,2),(12,3),(12,4),(12,9),(12,10),(13,0),
                        (13,1),(13,2),(13,3),(13,4),(13,9),(13,10),(14,0),(14,1),(14,2),(14,3),(14,4),(14,9),(14,10),(15,9),
                        (15,10),(16,9),(16,10),(17,9),(17,10),(17,17),(17,18),(17,19),(18,9),(18,10),(18,18),(18,19),(19,9),(19,10)]
        for i in self.obstacles : 
            self.grid[i]=1
        self.final_states={(3,3,DOWN):50, (4,11,LEFT):100, (5,14,RIGHT):100,(15,2,UP):200, (19,19,UP):999}
        self.values= np.array(create_matrix(self.width, self.height,[-1,-1,-1,-1,-1]))        
            
        self.current_location = (0,0)     
        self.first_location=(0,0)
        for transition, reward in self.final_states.items():
            self.values[(transition[0],transition[1])][transition[2]]=reward
        self.actions = [UP,DOWN,LEFT,RIGHT,STAY]
              
    def make_step(self, action):
        last_location = self.current_location       
        reward = self.values[(last_location[0],last_location[1])][action]
        if action == UP:
            if last_location[0] != 0 and self.grid[last_location[0]-1][ last_location[1]]==0 :
                self.current_location = ( last_location[0] - 1, last_location[1])        
        elif action == DOWN:
            if last_location[0] != self.height - 1 and self.grid[last_location[0]+1][last_location[1]]==0:
                self.current_location = ( last_location[0] + 1, last_location[1])
        elif action == LEFT:
            if last_location[1] != 0 and self.grid[last_location[0]][last_location[1]-1]==0:
                self.current_location = (last_location[0],last_location[1]-1)
        elif action == RIGHT:
            if last_location[1] != self.width - 1 and self.grid[last_location[0]][last_location[1]+1]==0:
                self.current_location = ( last_location[0], last_location[1] + 1)    
        elif action == STAY: self.current_location= last_location
        return reward, (last_location[0],last_location[1],action) in self.final_states.keys()

import pygame
import math
import numpy as np   

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4

class Graphique():

    def __init__(self, screen_size,cell_width, 
        cell_height, cell_margin,grid,rewards,init,table=np.zeros((0,0)),actions={}):
        
        
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.PURPLE=(108,2,119)
        self.WIDTH = cell_width
        self.HEIGHT = cell_height
        self.MARGIN = cell_margin
        self.color = self.WHITE
        
        pygame.init()
        pygame.font.init()
        self.size = (screen_size, screen_size)
        self.screen = pygame.Surface(self.size)
        
        self.font = pygame.font.SysFont('arial', 16)

        pygame.display.set_caption("Classe d'états")    
        self.clock = pygame.time.Clock()

        self.grid = grid
        self.init=init
        self.table=table
        self.actions=actions
        self.screen.fill(self.BLACK)

        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if [row, col] == self.init:
                    self.color = self.BLUE
                elif grid[row][col] == -1.:
                    self.color =self.BLACK
                else:
                    self.color = self.WHITE
                pygame.draw.rect(self.screen,
					self.color,
					[(self.MARGIN + self.WIDTH)*col+self.MARGIN,
					(self.MARGIN + self.HEIGHT)*row+self.MARGIN,
					self.WIDTH,
					self.HEIGHT])
                
        
        
        if table.size >0:
            for row in range(len(table)):
                for col in range(len(table[0])): 
                    if grid[row][col]!=-1:
                        self.color=(255,max(0,255*(1-table[(row,col)])),max(0,255*(1-table[row,col])))
                        pygame.draw.rect(self.screen,
					                 self.color,[(self.MARGIN + self.WIDTH)*col+self.MARGIN,
					                 (self.MARGIN + self.HEIGHT)*row+self.MARGIN,self.WIDTH,self.HEIGHT])


        max_value=max(rewards.values())
        number_keys=len(rewards)
        for key,value in rewards.items():
            if key[2] != STAY:
                position_correction={UP:(15,25),DOWN:(35,25),LEFT:(25,15),RIGHT:(25,35)}
                y=50*key[0]+position_correction[key[2]][0]
                x=50*key[1]+position_correction[key[2]][1]
                angles={UP:0,DOWN:180,RIGHT:270,LEFT:90}
                angle=angles[key[2]]
                if value==max_value : self.DrawArrow(x, y,self.GREEN,angle)
                else : self.DrawArrow(x, y,self.PURPLE,angle)
                """text_corrections={UP:(18,8),DOWN:(-35,8),LEFT:(20,0),RIGHT:(20,0)}
                label=self.font.render('+'+str(value),1,self.BLACK)
                x=50*key[0]+text_corrections[key[2]][0]
                y=50*key[1]+text_corrections[key[2]][1]
                self.screen.blit(label, (y,x))"""
            if key[2]==STAY and number_keys<5:
                x=50*(key[1])+27.5
                y=50*(key[0])+27.5
                if value==max_value : pygame.draw.circle(self.screen, self.GREEN, (x,y), 22)
                else : pygame.draw.circle(self.screen, self.PURPLE, (x,y), 15)
            elif key[2]==STAY:
                x=50*(key[1])+27.5
                y=50*(key[0])+27.5
                if value==max_value : pygame.draw.circle(self.screen, self.GREEN, (x,y), 15)
                else : pygame.draw.circle(self.screen, self.PURPLE, (x,y), 8)
        pygame.draw.rect(self.screen, self.BLUE, pygame.Rect(50*(self.init[1])+14, 50*(self.init[0])+14, 25, 25))
                            
        
        
        if actions!={}:
            for (row,col),value in actions.items():
                    if grid[row][col]!=-1:
                        best_action=value
                        x,y=50*col+27.5,50*row+27.5
                        if best_action==STAY:
                            pygame.draw.circle(self.screen, self.BLACK, (x,y), 8)
                        if best_action != STAY : 
                            angles={UP:0,DOWN:180,RIGHT:270,LEFT:90}
                            correction={UP:(0,5),DOWN:(0,-5),LEFT:(5,0),RIGHT:(-5,0)}
                            x+=correction[best_action][0]
                            y+=correction[best_action][1]
                            self.DrawArrow(x, y,self.BLACK,angles[best_action])
                            
    def show(self):
        pygame.display.flip()

    def DrawArrow(self,x,y,color,angle=0):
        def rotate(pos, angle):	
            cen = (x,y)
            angle *= -(math.pi/180)
            cos_theta = math.cos(angle)
            sin_theta = math.sin(angle)
            ret = ((cos_theta * (pos[0] - cen[0]) - sin_theta * (pos[1] - cen[1])) + cen[0],
                   (sin_theta * (pos[0] - cen[0]) + cos_theta * (pos[1] - cen[1])) + cen[1])
            return ret		
        p0 = rotate((0+x,-8+y), angle+90)
        p1 = rotate((0+x,8+y), angle+90)
        p2 = rotate((16+x,0+y), angle+90)
        pygame.draw.polygon(self.screen, color, [p0,p1,p2])

class Transition():

    def __init__(self, init, titre,transitions,walls,action,screen_size=450,cell_width=130, cell_height=130, cell_margin=15):
        self.WHITE=(255,255,255)
        self.BLACK=(0,0,0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.WIDTH = cell_width
        self.HEIGHT = cell_height
        self.MARGIN = cell_margin

        pygame.init()
        pygame.font.init()
        self.size = (screen_size, screen_size)
        self.screen = pygame.display.set_mode(self.size)

        self.font = pygame.font.SysFont('arial', 20)

        pygame.display.set_caption(titre)    

        self.transitions=transitions
        self.walls=walls
        self.screen.fill(self.BLACK)
        self.init=init

        for row in range(3):
            for col in range(3):
                if walls[(row+self.init[0]-1,col+self.init[1]-1)] == 1:
                    self.color =self.BLACK
                else:
                    self.color = self.WHITE
                pygame.draw.rect(self.screen,
					self.color,
					[(self.MARGIN + self.WIDTH)*col+self.MARGIN,
					(self.MARGIN + self.HEIGHT)*row+self.MARGIN,
					self.WIDTH,
					self.HEIGHT])
                y=150*row+45
                x=150*col+50
                x_row, y_col =row+self.init[0]-1, col+self.init[1]-1
                if (x_row,y_col) in self.transitions.keys():
                    label=self.font.render(str(round(self.transitions[(x_row,y_col)]*100,3))+" %",1,self.BLACK)
                    self.screen.blit(label,(x,y))
                if (row,col)==(1,1):
                        position={UP:(0,0),DOWN:(0,-35),LEFT:(10,-20),RIGHT:(-10,-20)}
                        x=220+position[action][0]
                        y=275+position[action][1]
                        angles={UP:0,DOWN:180,RIGHT:270,LEFT:90}
                        angle=angles[action]
                        self.DrawArrow(x, y,self.BLUE,angle)
    def DrawArrow(self,x,y,color,angle=0):
        def rotate(pos, angle):	
            cen = (5+x,0+y)
            angle *= -(math.pi/180)
            cos_theta = math.cos(angle)
            sin_theta = math.sin(angle)
            ret = ((cos_theta * (pos[0] - cen[0]) - sin_theta * (pos[1] - cen[1])) + cen[0],
                   (sin_theta * (pos[0] - cen[0]) + cos_theta * (pos[1] - cen[1])) + cen[1])
            return ret		
        p0 = rotate((0+x,-20+y), angle+90)
        p1 = rotate((0+x,20+y), angle+90)
        p2 = rotate((40+x,0+y), angle+90)
        pygame.draw.polygon(self.screen, color, [p0,p1,p2])


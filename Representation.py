import pygame
import math
from Complexworld import ComplexState
from Gridworld import State
import numpy as np   

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4

class Graphique():

    def __init__(self, screen_size,cell_width, 
        cell_height, cell_margin,grid,finals,init,table=np.zeros((0,0))):
        
        
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
        self.screen = pygame.display.set_mode(self.size)
        
        self.font = pygame.font.SysFont('arial', 16)

        pygame.display.set_caption("Classe d'états")    
        self.clock = pygame.time.Clock()

        self.grid = grid
        self.init=init
        self.table=table

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
        max_value=max(finals.values())
        for key,value in finals.items():
            if key[2] != STAY:
                labels={UP:(10,20),DOWN:(-10,20),LEFT:(25,-7),RIGHT:(25,37)}
                y=50*key[0]+labels[key[2]][0]
                x=50*key[1]+labels[key[2]][1]
                angles={UP:0,DOWN:180,RIGHT:270,LEFT:90}
                angle=angles[key[2]]
                if value==max_value : self.DrawArrow(x, y,self.GREEN,angle)
                else : self.DrawArrow(x, y,self.PURPLE,angle)
                labels2={UP:(18,8),DOWN:(-35,8),LEFT:(20,0),RIGHT:(20,0)}
                label=self.font.render('+'+str(value),1,self.BLACK)
                x=50*key[0]+labels2[key[2]][0]
                y=50*key[1]+labels2[key[2]][1]
                self.screen.blit(label, (y,x))
            if key[2]==STAY:
                x=50*(key[1])+25
                y=50*(key[0])+25
                if value==max_value : pygame.draw.circle(self.screen, self.GREEN, (x,y), 15)
                else : pygame.draw.circle(self.screen, self.PURPLE, (x,y), 15)
        pygame.draw.circle(self.screen, self.BLUE, (50*(self.init[1])+25, 50*(self.init[0])+25), 15) 
        
    def show(self):
        pygame.display.flip()

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


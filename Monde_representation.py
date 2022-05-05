import pygame


class Monde():

    def __init__(self, grid,recompenses=[],screen_size=500,cell_width=44.8, 
        cell_height=44.8, cell_margin=5):       
        
        pygame.init()
        pygame.font.init()
        
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.ORANGE=(255,165,0)
        self.GREEN=(0,255,0)
        self.PURPLE=(108,2,119)
        self.WIDTH = cell_width
        self.HEIGHT = cell_height
        self.MARGIN = cell_margin
        self.color = self.WHITE
        self.size = (screen_size, screen_size)
        self.screen = pygame.display.set_mode(self.size)

        pygame.display.set_caption("Monde généré aléatoirement")
        
        self.font = pygame.font.SysFont('arial', 20)

        self.clock = pygame.time.Clock()

        self.grid = grid

        self.screen.fill(self.BLACK)

        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row,col] == -1:
                    self.color =self.BLACK
                elif grid[row,col]==-2:
                    self.color = self.BLUE
                else : self.color= self.WHITE 
                if len(recompenses)==0 : 
                    if grid[row,col]==1:
                        self.color=self.BLUE
                    if grid[row,col]>=6 and grid[row,col]<=10:
                        self.color=self.ORANGE
                    elif grid[row,col]>15 and grid[row,col]<50:
                        self.color=self.RED
                else :
                    max_value=max(recompenses)
                    if grid[row,col] in recompenses:
                        if grid[row,col]==max_value:self.color=self.RED
                        else : self.color=self.ORANGE
                pygame.draw.rect(self.screen,
					self.color,
					[(self.MARGIN + self.WIDTH)*col+self.MARGIN,
					(self.MARGIN + self.HEIGHT)*row+self.MARGIN,
					self.WIDTH,
					self.HEIGHT])
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                y=50*col+12
                x=50*row+25
                if len(recompenses)>0:
                    if grid[row,col] not in [0,-2]:
                        label=self.font.render(str(grid[row,col]),1,self.BLACK)
                        self.screen.blit(label,(y,x))
                else : 
                    if grid[row,col] not in [0,-2]:
                        label=self.font.render(str(int(grid[row,col])),1,self.BLACK)
                        self.screen.blit(label,(y,x))
        
        def show(self):
            pygame.display.flip()



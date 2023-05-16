import pygame
import pygame.locals
import time

pygame.init()

fenetre = pygame.display.set_mode((500, 500))

rond = pygame.image.load("rond.svg").convert_alpha()
noir = pygame.image.load("noir.png").convert()

class Rond:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def affiche(self):
        fenetre.blit(rond, (self.x, self.y))

    def tombe(self):
        if (self.y < 450): self.y += 1

    def deplace(self, dx, dy):
        self.x += dx
        self.y += dy

    def vise(self, x, y):
        self.x = self.x + (x-self.x)//20
        self.y = self.y + (y-self.y)//20

rond1 = Rond(100, 100)

rond1.affiche()
pygame.display.flip()

continuer = True
while continuer:
    """for event in pygame.event.get():
        if event.type == pygame.locals.KEYDOWN:
            if event.key == pygame.locals.K_SPACE:
                continuer = False
            elif event.key == pygame.locals.K_z:
                rond1.deplace(0, -20)
            elif event.key == pygame.locals.K_q:
                rond1.deplace(-20, 0)
            elif event.key == pygame.locals.K_s:
                rond1.deplace(0, 20)
            elif event.key == pygame.locals.K_d:
                rond1.deplace(20, 0)
    """
    x, y = pygame.mouse.get_pos()
    rond1.vise(x, y)
    
    fenetre.blit(noir, (0, 0))
    rond1.affiche()
    pygame.display.flip()
    time.sleep(0.016)

pygame.quit()
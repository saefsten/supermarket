import numpy as np
import pandas as pd
from anders_sm_config import TILE_SIZE, OFS

class Customer:

    def __init__(self, terrain_map, customerid, image, location, x, y):

        self.terrain_map = terrain_map
        self.customerid = customerid
        self.image = image
        self.x = x
        self.y = y
        self.location = location
        self.locations = ['checkout','dairy','drinks','fruit','spices']
        self.content = [list(row) for row in terrain_map.split('\n')]
        self.matrix = pd.read_csv('transition_matrix.csv', sep=';',index_col=['location'])

    def change_location(self):
        '''where does the customer go next'''
        new_location = np.random.choice(self.locations, p=self.matrix.loc[self.location])
        self.location = new_location

    def cust_draw(self, frame):
        xpos = OFS + self.x * TILE_SIZE
        ypos = OFS + self.y * TILE_SIZE
        frame[ypos : ypos + self.image.shape[0], xpos : xpos + self.image.shape[1]] = self.image
        # overlay the Customer image / sprite onto the frame
    
    def move(self, direction):
        x_new = self.x #initial position
        y_new = self.y
        if direction == 'up':
            y_new -= 1
        elif direction == 'down':
            y_new += 1
        elif direction == 'left':
            x_new -= 1
        elif direction == 'right':
            x_new += 1
        else:
            pass

        if self.content[y_new][x_new] == '.':
            self.x = x_new
            self.y = y_new
        else:
            pass
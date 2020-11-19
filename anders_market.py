import pandas as pd
import numpy as np
from cv2 import cv2
import time
from anders_supermarket_class import Supermarket
from anders_customer import Customer
from anders_sm_config import MARKET, tiles, matrix, TILE_SIZE, OFS

class SupermarketMap:
    """Visualizes the supermarket background"""

    def __init__(self, layout, tiles):
        """
        layout : a string with each character representing a tile = MARKET
        tile   : a numpy array containing the tile image
        """
        self.tiles = tiles
        self.contents = [list(row) for row in layout.split('\n')]
        self.xsize =  len(self.contents[0])
        self.ysize = len(self.contents)
        self.image = np.zeros((self.ysize * TILE_SIZE, self.xsize * TILE_SIZE, 3), dtype=np.uint8)
        self.prepare_map()

    def get_tile(self, char):
        """returns the array for a given tile character"""
        if char == '#':
            return self.tiles[0:32, 0:32]
        elif char == 'b': #banana
            return self.tiles[0:32,4*32:5*32]
        elif char == 'B': #eggplant
            return self.tiles[1*32:2*32,11*32:12*32]
        elif char == 'w': #watermelon
            return self.tiles[3*32:4*32, 4*32:5*32]
        elif char == 'W': #pinapple
            return self.tiles[5*32:6*32, 4*32:5*32]
        elif char == 'e': #egg
            return self.tiles[7*32:8*32, 11*32:12*32]
        elif char == 'E': #ice cream
            return self.tiles[6*32:7*32, 12*32:13*32]
        elif char == 'f': #ice cream cone
            return self.tiles[0*32:1*32, 13*32:14*32]
        elif char == 's': #spices
            return self.tiles[4*32:5*32, 9*32:10*32]
        elif char == 'S': #spices
            return self.tiles[5*32:6*32, 9*32:10*32]
        elif char == 'z': #spices
            return self.tiles[6*32:7*32, 9*32:10*32]
        elif char == 'u': #beer
            return self.tiles[6*32:7*32, 13*32:14*32]
        elif char == 'U': #cocktail
            return self.tiles[3*32:4*32, 13*32:14*32]
        elif char == 'G':
            return self.tiles[7*32:8*32, 3*32:4*32]
        elif char == 'C':
            return self.tiles[2*32:3*32, 8*32:9*32]
        elif char == 'c':
            return self.tiles[1*32:2*32, 8*32:9*32]
        elif char == 'q':
            return self.tiles[3*32:4*32, 8*32:9*32]
        else:
            return self.tiles[32:64, 64:96]

    def prepare_map(self):
        """prepares the entire image as a big numpy array"""
        for y, row in enumerate(self.contents): #self.contents is a list of list of MARKET
            for x, tile in enumerate(row):
                bm = self.get_tile(tile) #find the image belonging to the tile in MARKET
                self.image[y * TILE_SIZE:(y+1)*TILE_SIZE,
                      x * TILE_SIZE:(x+1)*TILE_SIZE] = bm #set that postition in MARKET as that image

    def draw(self, frame, offset=OFS):
        """
        draws the image into a frame
        offset pixels from the top left corner
        """
        frame[OFS:OFS+self.image.shape[0], OFS:OFS+self.image.shape[1]] = self.image

    def write_image(self, filename):
        """writes the image into a file"""
        cv2.imwrite(filename, self.image)


if __name__ == "__main__":

    background = np.zeros((700, 1000, 3), np.uint8)
    market = SupermarketMap(MARKET, tiles)
    supermarket = Supermarket()
    customers_list = []
    
    me_pic = tiles[7*32:8*32,15*32:16*32]
    me = Customer(MARKET, 0, me_pic, 'entrance', 14, 10)

    while True:
        frame = background.copy()
        market.draw(frame)
        me.cust_draw(frame)
        supermarket.remove_existing_customers(customers_list)
        supermarket.next_minute(customers_list)
        supermarket.add_new_customers()

        for cust in customers_list:
            cust.cust_draw(frame)

        cv2.imshow('frame', frame)

        print(supermarket.print_customers())

        time.sleep(0.5)
        key = chr(cv2.waitKey(1) & 0xFF)
        if key == 'q':
            break
        if key == 'i':
            me.move('up')
        if key == 'm':
            me.move('down')
        if key == 'j':
            me.move('left')
        if key == 'k':
            me.move('right')


    cv2.destroyAllWindows()

    # market.write_image("supermarket.png")

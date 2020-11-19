import numpy as np
import pandas as pd
import cv2
import time
from routefinder import find_path

TILE_SIZE = 32
OFS = 50

MARKET = """
##################
##..............##
##..##..##..##..##
#B..#S..#D..#F..##
##..##..##..##..##
#B..#S..#D..#F..##
##..##..##..##..##
##...............#
##..C#..C#..C#...#
##..##..##..##...#
##...............#
##############GG##
""".strip()

MARKET2 = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])

class SupermarketMap:
    """Visualizes the supermarket background"""

    def __init__(self, layout, tiles):
        """
        layout : a string with each character representing a tile
        tile   : a numpy array containing the tile image
        """
        self.tiles = tiles
        self.contents = [list(row) for row in layout.split("\n")]
        self.xsize = len(self.contents[0])
        self.ysize = len(self.contents)
        self.image = np.zeros(
            (self.ysize * TILE_SIZE, self.xsize * TILE_SIZE, 3), dtype=np.uint8
        )
        self.prepare_map()

    def get_tile(self, char):
        """returns the array for a given tile character"""
        if char == "#":
            return self.tiles[0:32, 0:32]
        elif char == "G":
            return self.tiles[7 * 32 : 8 * 32, 3 * 32 : 4 * 32]
        elif char == "C":
            return self.tiles[2 * 32 : 3 * 32, 8 * 32 : 9 * 32]
        elif char == "B":
            return self.tiles[3 * 32 : 4 * 32, 13 * 32 : 14 * 32]
        elif char == "S":
            return self.tiles[6 * 32 : 7 * 32, 9 * 32 : 10 * 32]
        elif char == "D":
            return self.tiles[6 * 32 : 7 * 32, 3 * 32 : 4 * 32]
        elif char == "F":
            return self.tiles[0 : 32, 4 * 32 : 5 * 32]
        else:
            return self.tiles[32:64, 64:96]

    def prepare_map(self):
        """prepares the entire image as a big numpy array"""
        for y, row in enumerate(self.contents):
            for x, tile in enumerate(row):
                bm = self.get_tile(tile)
                self.image[
                    y * TILE_SIZE : (y + 1) * TILE_SIZE,
                    x * TILE_SIZE : (x + 1) * TILE_SIZE,
                ] = bm

    def draw(self, frame, offset=OFS):
        """
        draws the image into a frame
        offset pixels from the top left corner
        """
        frame[
            OFS : OFS + self.image.shape[0], OFS : OFS + self.image.shape[1]
        ] = self.image

    def write_image(self, filename):
        """writes the image into a file"""
        cv2.imwrite(filename, self.image)



class Customer:

    def __init__(self, id, location, terrain_map, image, x, y):
        self.id = id
        self.location = location

        self.terrain_map = terrain_map
        self.image = image
        self.x = x
        self.y = y
        
        self.goal = [0,0]

    def draw(self, frame):
        xpos = OFS + self.x * TILE_SIZE
        ypos = OFS + self.y * TILE_SIZE
        frame[ypos: ypos + 32, xpos: xpos + 32] = self.image

        # overlay the Customer image / sprite onto the frame
        
    def moveonmap (self, direction):
        if direction == 'u':
            if self.terrain_map.contents[self.y - 1][self.x] == '.':
                self.y -= 1
        if direction == 'd':
            if self.terrain_map.contents[self.y + 1][self.x] == '.':
                self.y += 1
        if direction == 'r':
            if self.terrain_map.contents[self.y][self.x + 1] == '.':
                self.x += 1
        if direction == 'l':
            if self.terrain_map.contents[self.y][self.x - 1] == '.':
                self.x -= 1

    def __repr__(self):
        return f'Customer ({self.id}, {self.location})'

    def is_active(self):
        '''is the customer still shopping?'''
        if self.location == 'checkout':
            return False
        else:
            return True

    def change_location(self):
        '''where does the customer go next'''
        new_location = np.random.choice(locations, p=matrix.loc[self.location])
        self.location = new_location



class Supermarket:
    """manages multiple Customer instances that are currently in the market.
    """

    def __init__(self, marketmap):        
        # a list of Customer objects
        self.customers = []
        self.minutes = 419
        self.last_id = 0
        self.marketmap = marketmap
        
    def __repr__(self):
        return f'T: {self.minutes}, Customers: {self.customers}'

    def get_time(self):
        """current time in HH:MM format,
        """
        hours = str(self.minutes//60)
        if len(hours) == 1:
            hours = '0' + hours
        minutes = str(self.minutes % 60)
        if len(minutes) == 1:
            minutes = '0' + minutes
        return f'{hours}:{minutes}'

    def print_customers(self):
        """print all customers with the current time and id in CSV format.
        """
        currentstatus = []
        for element in self.customers:
            currentstatus.append([self.get_time(),element.id,element.location])
        return currentstatus            

    def next_minute(self, frame):
        """propagates all customers to the next state.
        """
        self.minutes += 1
#        frame = np.zeros((700, 1000, 3), np.uint8)
        self.marketmap.draw(frame)        
        for customer in self.customers:
            oldlocation = customer.location
            customer.change_location()            
            newlocation = customer.location
            if oldlocation != newlocation:
                if newlocation != 'checkout':
                    customer.goal = waitinglocation[newlocation]
                else:
                    customer.goal = waitinglocation[oldlocation+'checkout']                    
#                customer.x, customer.y = waitinglocation[newlocation]
                print(f'Customer {customer.id} moves from {oldlocation} to: {newlocation}.')
            else:   
                print(f'Customer {customer.id} is taking some more time at: {oldlocation}.')
            customer.draw(frame)
#        cv2.imshow("frame", frame)
        
    def add_new_customers(self):
        """randomly creates new customers.
        """
        if np.random.rand() < .33:
            self.last_id += 1
            customerid = str(self.last_id)
#            self.customers.append(Customer(customerid,'entrance'))
            self.customers.append(Customer(str(len(self.customers) + 1),'entrance', marketmap, tiles[7 * 32 : 8 * 32, 2 * 32 : 3 * 32],15,10))

#            print(f'Customer {customerid} entered the supermarket.')

    def remove_existing_customers(self):
        """removes every customer that is not active any more.
        """
        j = 0
        while j < len(self.customers):
            if self.customers[j].location == 'checkout':
                self.customers.remove(self.customers[j])
            else:
                j += 1


if __name__ == "__main__":

    locations = ['checkout','dairy','drinks','fruit','spices']
    waitinglocation = {}
    waitinglocation['dairy'] = [10,4]
    waitinglocation['drinks'] = [2,4]
    waitinglocation['fruit'] = [14,4] 
    waitinglocation['spices'] = [6,4]
    waitinglocation['drinkscheckout'] = [4,7]
    waitinglocation['spicescheckout'] = [4,7]
    waitinglocation['dairycheckout'] = [8,7]
    waitinglocation['fruitcheckout'] = [12,7]    
    matrix = pd.read_csv('transition_matrix.csv', sep=';',index_col=['location'])    

    background = np.zeros((700, 1000, 3), np.uint8)
    tiles = cv2.imread("tiles.png")

    marketmap = SupermarketMap(MARKET, tiles)

    hmframe = np.zeros((700, 1000, 3), np.uint8)
    cv2.imshow("frame", hmframe)


    possible_moves = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

    doodl = Supermarket(marketmap)
    
    df = pd.DataFrame(columns=['Time','CustomerID','Location'])
    while doodl.minutes < 22 * 60:
        frame = background.copy()
        doodl.next_minute(frame)
        print(f'Current time is {doodl.get_time()}.')
        doodl.add_new_customers()
        data = doodl.print_customers()

        moving = 1
        while moving == 1:
            doodl.marketmap.draw(frame)
            moving = 0
            for element in doodl.customers:
                if element.goal != [0,0]:
                    moving = 1
                    goal1 = element.goal.copy()
                    path = find_path(MARKET2, [element.y, element.x], goal1, possible_moves)
                    element.y = path[1][0]
                    element.x = path[1][1] 

                    if element.goal == [element.x, element.y]:
                        element.goal = [0,0]
                element.draw(frame)

            cv2.imshow("frame", frame)
            k = cv2.waitKey(100) 

        for element in data:
            df.loc[len(df)] = element
        doodl.remove_existing_customers()

    print('The supermarket is closing. All remaining customers rush to the checkout!')
    df.to_csv('simulation.csv')

    cv2.destroyAllWindows()

    market.write_image("supermarket.png")




from cv2 import cv2
import pandas as pd
import numpy as np

tiles = cv2.imread('tiles.png')
#customer_pic = tiles[5*32:6*32,15*32:16*32]

MARKET = """
##################
##..............##
#w..Be..eU..us..s#
#w..Be..eU..us..s#
#w..be..eU..uz..z#
#W..bE..fU..uz..S#
#W..bE..fU..uS..S#
##...............#
##..C#..c#..q#...#
##..##..##..##...#
##...............#
##############GG##
""".strip()

matrix = pd.read_csv('transition_matrix.csv', sep=';',index_col=['location'])

TILE_SIZE = 32
OFS = 50

def pick_customer():
    pic_dict = {
    'green_fish' : tiles[5*32:6*32,15*32:16*32],
    'darkblue_fish' : tiles[4*32:5*32,15*32:16*32],
    'blue_fish' : tiles[6*32:7*32,15*32:16*32],
    'butterfly' : tiles[5*32:6*32,14*32:15*32],
    'insect' : tiles[3*32:4*32,14*32:15*32],
    'ghost' : tiles[7*32:8*32,0*32:1*32],
    'shell' : tiles[7*32:8*32,14*32:15*32],
    'pacman' : tiles[3*32:4*32,0*32:1*32]
    }
    pic_list = ['green_fish', 'darkblue_fish', 'blue_fish', 'butterfly', 
        'insect', 'ghost', 'shell', 'pacman']
    return pic_dict[np.random.choice(pic_list)]

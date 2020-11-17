import pandas as pd
import numpy as np
import time

locations = ['checkout','dairy','drinks','fruit','spices']

matrix = pd.read_csv('transition_matrix.csv', sep=';',index_col=['location'])


class Customer:

    def __init__(self,id,location):
        self.id = id
        self.location = location

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

z = 5
customers = []
for customer in range(1,z):
    customers.append(Customer(customer,'entrance'))

t = 1
while True:
    # Change location or remove from list
    for customer in customers:
        if customer.is_active():
            print('before '+str(customer))
            customer.change_location()
            print('after '+str(customer))
        else:
            customers.remove(customer)
            del customer
    
    # Add new customer every 3rd round
    if t%3 == 0:
        customers.append(Customer(z,'entrance'))
        z += 1

    # Print customer list
    if customers == []:
        print('Store is empty!')
    else:
        for customer in customers:
            print(customer)
    print('\n')
    t += 1
    time.sleep(3)
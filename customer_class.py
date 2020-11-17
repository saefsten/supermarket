import pandas as pd
import numpy as np
import time

locations = ['checkout','dairy','drinks','fruit','spices']
data = {
    'dairy': [0.35,0.0,0.24,0.2,0.21],
    'drinks': [0.53, 0.03, 0.0, 0.23, 0.21],
    'entrance':[0.0,0.28,0.16,0.36,0.2],
    'fruit':[0.52,0.23,0.13,0.0,0.12],
    'spices':[0.24,0.31,0.29,0.16,0.0]
    }
matrix = pd.DataFrame.from_dict(data, orient='index', columns=locations)

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
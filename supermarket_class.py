"""
Supermarket simulator class.
"""

import numpy as np
import pandas as pd
from customer_class import Customer


class Supermarket:
    """manages multiple Customer instances that are currently in the market.
    """

    def __init__(self):        
        # a list of Customer objects
        self.customers = []
        self.minutes = 419
        self.last_id = 0

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

    def next_minute(self):
        """propagates all customers to the next state.
        """
        self.minutes += 1
        for customer in self.customers:
            oldlocation = customer.location
            print(f'Customer {customer.id} moves from {oldlocation} to:')
            customer.change_location()            
            newlocation = customer.location
            print(f' {newlocation}.')

    
    def add_new_customers(self):
        """randomly creates new customers.
        """
        if np.random.rand() < .33:
            self.last_id += 1
            customerid = str(self.last_id)
            self.customers.append(Customer(customerid,'entrance'))
            print(f'Customer {customerid} entered the supermarket.')

    def remove_existing_customers(self):
        """removes every customer that is not active any more.
        """
        j = 0
        while j < len(self.customers):
            if self.customers[j].location == 'checkout':
                self.customers.remove(self.customers[j])
            else:
                j += 1
                


doodl = Supermarket()
df = pd.DataFrame(columns=['Time','CustomerID','Location'])
while doodl.minutes < 22 * 60:
    doodl.next_minute()
    print(f'Current time is {doodl.get_time()}.')
    doodl.add_new_customers()
    data = doodl.print_customers()
    for element in data:
        df.loc[len(df)] = element
    doodl.remove_existing_customers()
print('The supermarket is closing. All remaining customers, rush to the checkout!')
df.to_csv('simulation.csv')
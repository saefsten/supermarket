import numpy as np
from anders_customer import Customer
from cv2 import cv2
from anders_sm_config import MARKET, tiles, pick_customer

fruit_section = {'x':[2,3], 'y':[2,3,4,5,6]}
dairy_section = {'x':[6,7], 'y':[2,3,4,5,6]}
drinks_section = {'x':[10,11], 'y':[2,3,4,5,6]}
spices_section = {'x':[14,15], 'y':[2,3,4,5,6]}
checkout_section = {'x':[3,7,11]}

class Supermarket:
    """manages multiple Customer instances that are currently in the market.
    """

    def __init__(self):        
        # a list of Customer objects
        #self.customers_list = []
        self.minutes = 419
        self.last_id = 0

    def __repr__(self):
        return f'T: {self.minutes}, Customers: {self.customers_list}'          

    def next_minute(self,customers_list):
        """propagates all customers to the next state.
        """
        self.minutes += 1
        self.customers_list = customers_list
        for customer in self.customers_list:
            #customer_id = customer.customerid
            #oldlocation = customer.location
            #print(f'Customer {customer_id} moves from {oldlocation} to:')
            customer.change_location()            
            newlocation = customer.location
            #print(newlocation)
            if newlocation == 'dairy':
                customer.x = np.random.choice(dairy_section['x'])
                customer.y = np.random.choice(dairy_section['y'])
            elif newlocation == 'spices':
                customer.x = np.random.choice(spices_section['x'])
                customer.y = np.random.choice(spices_section['y'])
            elif newlocation == 'fruit':
                customer.x = np.random.choice(fruit_section['x'])
                customer.y = np.random.choice(fruit_section['y'])
            elif newlocation == 'drinks':
                customer.x = np.random.choice(drinks_section['x'])
                customer.y = np.random.choice(drinks_section['y'])
            elif newlocation == 'checkout':
                customer.x = np.random.choice(checkout_section['x'])
                customer.y = 8
            else:
                print('where am I?')
                pass

    def add_new_customers(self):
        """randomly creates new customers. Creates it at the entrance
        """
        
        if np.random.rand() < .33:
            self.last_id += 1
            ncustomerid = str(self.last_id)
            customer_pic = pick_customer()
            new_customer = Customer(MARKET, ncustomerid, customer_pic, 'entrance', 14, 10)
            self.customers_list.append(new_customer)
            #print(f'Customer {ncustomerid} entered the supermarket.')

    def remove_existing_customers(self, customers_list):
        """removes every customer that is not active any more.
        """
        self.customers_list = customers_list
        k = 0
        while k < len(self.customers_list):
            if self.customers_list[k].location == 'checkout':
                #leaving_id = self.customers_list[k].customerid
                #print(f'Customer {leaving_id} has left the supermarket.')
                self.customers_list.remove(self.customers_list[k])
            else:
                k += 1

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
        for element in self.customers_list:
            currentstatus.append([self.get_time(),element.customerid,element.location])
        return currentstatus  
#!/usr/bin/env python
# coding: utf-8

"""
Project Markov - EDA
@author: boti
"""

# Import pandas etc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os







def plotsections(df,dayofweek):
    totalcustomers = df['customer_no'].nunique()
    print(totalcustomers)
    print (f'Total number of customers on {dayofweek} : {str(totalcustomers)}')
    print (f'Total number of customers at each location on {dayofweek} :')
    print(df.groupby('location')['customer_no'].nunique())
#    print('Spices')
    df[df['location']=='spices'].resample('1h')['customer_no'].nunique().plot()
#    print('Fruit')
    df[df['location']=='fruit'].resample('1h')['customer_no'].nunique().plot()
#    print('Drinks')
    df[df['location']=='drinks'].resample('1h')['customer_no'].nunique().plot()
#    print('Dairy')
    df[df['location']=='dairy'].resample('1h')['customer_no'].nunique().plot()
    filename = 'plots/sections-' + dayofweek + '.jpg'
    plt.savefig(filename)
    plt.cla()

    df[df['location']=='checkout'].resample('1h')['customer_no'].nunique().plot()
    filename = 'plots/checkout-' + dayofweek + '.jpg'
    plt.savefig(filename)
    plt.cla()

    finalcheckoutcustomers = totalcustomers - df[df['location']=='checkout']['customer_no'].count()
    print(f'An additional {finalcheckoutcustomers} cutomers check out at 22:00 on {dayofweek}.')
    


#os.chdir('/home/boti/Spiced/git-repos/stochastic-sage-student-code/project_08/')

monday = pd.read_csv('./monday.csv',sep=';',index_col='timestamp', parse_dates=True)
tuesday = pd.read_csv('./tuesday.csv',sep=';',index_col='timestamp', parse_dates=True)
wednesday = pd.read_csv('./wednesday.csv',sep=';',index_col='timestamp', parse_dates=True)
thursday = pd.read_csv('./thursday.csv',sep=';',index_col='timestamp', parse_dates=True)
friday = pd.read_csv('./friday.csv',sep=';',index_col='timestamp', parse_dates=True)
plotsections(monday,'Monday')
plotsections(tuesday,'Tuesday')
plotsections(wednesday,'Wednesday')
plotsections(thursday,'Thursday')
plotsections(friday,'Friday')


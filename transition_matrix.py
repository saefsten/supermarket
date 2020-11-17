import pandas as pd 
import numpy as np 

# list of the data files:
days = ["monday.csv","tuesday.csv","wednesday.csv","thursday.csv","friday.csv"]

# creating an empty dataframe for listing all the customer walks:
customer_walks = pd.DataFrame(columns=["timestamp","customer_no","location","next_location"])

for day in days:
    # loop through the five days to append all customer walks
    df = pd.read_csv(day, sep=";")

    for i in range(df["customer_no"].max()):
        # loop through all customers of each day
        customer_walk = df[df["customer_no"]==(i+1)].copy()
        # here we interpolate by the minute to record when people stay at a location longer:
        customer_walk["timestamp"] = pd.to_datetime(customer_walk["timestamp"])
        customer_walk.set_index("timestamp", inplace=True)
        customer_walk = customer_walk.resample('T').bfill() #"T" resamples by minute
        # record the movement in "customer_walk"-dataframe with a new column "next_location":
        customer_walk["next_location"]=customer_walk["location"].shift(-1)
        customer_walk = customer_walk.reset_index()
        # if next location is missing it is set to "checkout" - for covering the last customers where checkout is missing:
        customer_walk = customer_walk.fillna("checkout")

        # first the entry into the store and the first location of each customer is added, then the rest of the walk:
        customer_walks = customer_walks.append({"timestamp":customer_walk["timestamp"].iloc[0], 
                                            "customer_no":customer_walk["customer_no"].iloc[0], 
                                            "location":"entrance", 
                                            "next_location":customer_walk["location"].iloc[0]},ignore_index=True)
        customer_walk["timestamp"]=customer_walk["timestamp"].shift(-1) #setting the timestamp to the time of location-change
        customer_walks = customer_walks.append(customer_walk)

# remove lines where both location and next_location are "checkout":
customer_walks = customer_walks[~((customer_walks["location"]=="checkout")&(customer_walks["next_location"]=="checkout"))]
# create timestamps for the last customers that we had to checkout "by force":
customer_walks = customer_walks.fillna(method="ffill")

# calculate the transition matrix from that:
ct = pd.crosstab(customer_walks["location"], customer_walks["next_location"], normalize=0)

# save the results in csv files:
customer_walks.to_csv("customer_walks.csv",sep=";")
ct.to_csv("transition_matrix.csv",sep=";")
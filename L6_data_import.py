#!/usr/bin/env python
# coding: utf-8

# ## Importing and manipulating data in Python

# There are many ways to import data in Python. The following shows two recommended examples for a csv file containing several columns with headers.

# ### (1) Numpy's loadtxt
# Can easily import arrays of data as a numpy array, but does not read the headers 

# In[16]:


import numpy as np

filename = 'Al alloy raw data T=1000s.csv'

dat1 = np.loadtxt(filename,skiprows=1,delimiter=',') # Cannot import first row of text, so skipping
print(f'type: {type(dat1)}, dimensions {dat1.shape}\n')
print(dat1)


# ### (2) Using pandas
# From https://pandas.pydata.org/:
# 
# *pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language*
# 
# Much more powerful approach, especially for more complex files. It can even read \*.xslx files directly.
# 
# Returns a pandas dataframe, an object similar to a table with headers. Compared to numpy arrays, columns may not contain only numbers.

# In[20]:


import pandas as pd

df = pd.read_csv(filename) # can also do read_excel()
print(f'type: {type(df)}, dimensions {df.shape}\n')

print(df) # dataframes printing gives a nice output


# Some example simple manipulations on pandas dataframes

# In[26]:


# This is the name of each row (by default 0,1,2,...)
print(df.index)

# This is the name of each columns (headers in the file we read)
print(df.columns)


# You can index and slice it the dataframe like a numpy array using "iloc"
# 

# In[40]:


print(df.iloc[0,1]) # returns a single element
print('\n')
print(df.iloc[2:4,1]) # return several rows at a pandas series
print('\n')
print(df.iloc[0,2:4]) # return several columns at a pandas series


# If you prefer, you can simply convert the dataframe to a numpy array
# 

# In[41]:


arr = df.to_numpy()
print(type(arr))
print(arr)


# ### Conclusion: recommended approach
# - Try to use pandas to import the data as it is more powerful and nicer
# - If you're familiar with pandas, you can do data manipulation using pandas dataframes
# - If not, convert the data to numpy

# ## Example: plotting imported data
# Below we plot the imported data. This gives further examples on capabilities of Matplotlib
# 

# In[75]:


import matplotlib.pyplot as plt

time=arr[:,0] # first column
ref = arr[:,1] # second column
peltier = arr[:,2] # second column
dat = arr[:,3:] # all columns from fourth with the data x=..
# dat is (5000,6) numpy
x = np.array([0.1,2.1,6.1,14.1,30.1,60.2])

fig, axs = plt.subplots(1,2) # creates a figure with an array of 1 by 2 subplots
axs[0].plot(time,ref) # first subplot
axs[0].set_title('Reference')
axs[0].set(xlabel='Time [ms]',ylabel='T [C]')
axs[1].plot(time,peltier)
axs[1].set_title('Peltier')
axs[1].set(xlabel='Time [ms]')


plt.figure(figsize=(12,8)) # creates a new figure
plt.plot(time,dat[:,0:6]) # plot all 6 data columns
plt.xlabel('Time [ms]')
plt.ylabel('T [C]')
plt.legend(['x=0.1cm','x=2.1cm','x=6.1cm','x=14.1cm','x=30.1cm','x=60.2cm'])


# ## Data manipulation
# numpy provides a number of functions for simple manipulation of the data, such as averaging, maximum/minimum, sum, etc...

# In[95]:


print(f'Average temperature in the x=0.1cm dataset (first column): {np.mean(dat[:,0])}\n')

# you can also call mean on a 2D array and specify the dimension along which the averaging is done
# For example, averaging over the rows, we obtain 6 values (average T in each column)
Tave0 = np.mean(dat,0) # This is a (6,) 1D array
print(Tave0.shape)
print(f'Average in each column: {Tave0}\n')
# Averaging over the columns, we obtain 5000 values (average T at a given time)
Tave1 = np.mean(dat,1) # This is a (5000,) 1D array
print(Tave1.shape)

# Similarly for function max
print(f'Maximum temperature in the x=2.1cm dataset (second column): {np.max(dat[:,1])}\n')

print(f'Maximum temperature at t=2ms (first row): {np.max(dat[0,:])}\n')

print(f'Maximum temperature across entire data set: {np.max(dat)}\n')

# Here specify maximum along one direction (rows) and returns a 1D array
print(f'Maximum temperatures for each column: {np.max(dat,0)}\n')


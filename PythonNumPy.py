# NumPy Array Slicing
import numpy as np 
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[1:5])
print()

arr = np.array([11, 12, 13, 14, 15, 16,17])
print(arr[4:])
print()

arr = np.array([12, 22, 32, 42, 52, 62, 72])
print(arr[:4])
print()
 
# Negative Slicing
arr = np.array([31, 32, 33, 34, 35, 36, 37])
print(arr[-3:-1])
print()

# STEP
arr = np.array([41, 42, 43, 44, 45, 46, 47])
print(arr[1:5:3])
print()

# Return every other element from the entire array
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[::2])
print()

# Slicing 2-D Arrays
# From the second element, slice elements from index 1 to index 4 (not included)
arr = np.array([[51, 52, 53, 54, 55], [56, 57, 58, 59, 510]])
print(arr[1, 1:4])
print()

# From both elements, return index 2
arr = np.array([[41, 42, 43, 44, 45], [46, 47, 48, 49, 410]])
print(arr[0:2, 2])
print()

# From both elements, slice index 1 to index 4 (not included), this will return a 2-D array
arr = np.array([[11, 12, 13, 14, 15], [16, 17, 18, 19, 110]])
print(arr[0:2, 1:4])
print()

# NumPy Data Types
arr = np.array([12, 23, 34, 45, 56])
print(arr.dtype)
print()

arr = np.array(['apple', 'banana', 'cherry'])
print(arr.dtype)
print()

# Creating Arrays With a Defined Data Type
arr = np.array([1, 2, 3, 4], dtype='S')
print(arr)
print(arr.dtype)
print()

# Create an array with data type 4 bytes integer
arr = np.array([1, 2, 3, 4], dtype='i4')
print(arr)
print(arr.dtype)
print()

# Converting Data Type on Existing Arrays
arr = np.array([1.2, 2.3, 3.4, 4.5, 5.6])
print(arr.dtype)
print()

newarr = arr.astype('i')
print(newarr)
print(newarr.dtype)
print()

# Change data type from float to integer by using int as parameter value
arr = np.array([1.21, 2.21, 3.21, 4.21])
print(arr.dtype)
print()

newarr = arr.astype(int)
print(newarr)
print(newarr.dtype)
print()

# Change data type from integer to boolean
arr = np.array([1, 0, 3, 4, 0, 2])
print(arr.dtype)
print()
newarr = arr.astype(bool)
print(newarr)
print(newarr.dtype)
print()

# NumPy Array Copy vs View
# COPY:
arr = np.array([11, 21, 31, 41, 51])
x = arr.copy()
arr[0] = 421

print(arr)
print()
print(x)
print()

# VIEW:
arr = np.array([10, 20, 30, 40, 50, 60, 70])
x = arr.view()
arr[0] = 402
print(arr)
print()
print(x)
print()

# Make Changes in the VIEW:
arr = np.array([11, 22, 32, 42, 52])
x = arr.view()
x[0] = 321
print(arr)
print()
print(x)
print()

# Check if Array Owns it's Data
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
y = arr.view()
print(x.base)
print()
print(y.base)
print()

# NumPy Array Shape
# Get the Shape of an Array
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(arr.shape)
print()

# Create an array with 5 dimensions using ndmin using a vector with values 1,2,3,4 and verify that last dimension has value 4
arr = np.array([21, 23, 34, 45, 4], ndmin = 5)
print(arr)
print('shape of array :', arr.shape)
print()

# NumPy Array Reshaping
# Reshape From 1-D to 2-D
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(4, 3)
print(newarr)
print()

# Returns Copy or View?
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print(arr.reshape(2, 4).base)
print()

# Unknown Dimension
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
newarr = arr.reshape(2, 2, -1)
print(newarr)
print()

# Flattening the arrays
arr = np.array([[1, 2, 3], [4, 5, 6]])
newarr = arr.reshape(-1)
print(newarr)
print()

# NumPy Array Iterating
# Iterating Arrays
arr = np.array([1, 2, 3])

for x in arr:
  print(x)
print()

# Iterating 2-D Arrays
arr = np.array([[1, 2, 3], [4, 5, 6]])
for x in arr:
  print(x)
print()

# Iterate on each scalar element of the 2-D array
arr = np.array([[1, 2, 3], [4, 5, 6]])
for x in arr:
  for y in x:
    print(y)
print()

# Iterating 3-D Arrays
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
for x in arr:
  print(x)
print()

# To return the actual values, the scalars, we have to iterate the arrays in each dimension
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
for x in arr:
  for y in x:
    for z in y:
      print(z)
print()

# Iterating Arrays Using nditer()
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
for x in np.nditer(arr):
  print(x)
print()

# Iterating Array With Different Data Types
arr = np.array([1, 2, 3])
for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
  print(x)
print()

# Iterating With Different Step Size
# Iterate through every scalar element of the 2D array skipping 1 element
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
for x in np.nditer(arr[:, ::2]):
  print(x)
print()

# Enumerated Iteration Using ndenumerate()
arr = np.array([1, 2, 3])
for idx, x in np.ndenumerate(arr):
  print(idx, x)
print()

# Enumerate on following 2D array's elements
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
for idx, x in np.ndenumerate(arr):
  print(idx, x)
print()

# Enumerate on following 3D array's elements
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
for idx, x in np.ndenumerate(arr):
  print(idx, x)
print()

# NumPy Joining Array
# Joining NumPy Arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))
print(arr)
print()

# Join two 2-D arrays along rows (axis=1)
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
arr = np.concatenate((arr1, arr2), axis=1)
print(arr)
print()

# Joining Arrays Using Stack Functions
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.stack((arr1, arr2), axis=1)
print(arr)
print()

# Stacking Along Rows
arr = np.hstack((arr1, arr2))
print(arr)
print()

# Stacking Along Columns
arr = np.vstack((arr1, arr2))
print(arr)
print()

# Stacking Along Height (depth)
arr = np.dstack((arr1, arr2))
print(arr)
print()

# NumPy Splitting Array
# Splitting NumPy Arrays
newarr = np.array_split(arr, 3)
print(newarr)
print()

arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 5)
print(newarr)
print()

# Split Into Arrays
newarr = np.array_split(arr, 3)
print(newarr[0])
print(newarr[1])
print(newarr[2])
print()

# Splitting 2-D Arrays
arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
newarr = np.array_split(arr, 3)
print(newarr)
print()

# Split the 2-D array into three 2-D arrays 
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3)
print(newarr)
print()

# The example below also returns three 2-D arrays, but they are split along the row (axis=1)
newarr = np.array_split(arr, 3, axis=1)
print(newarr)
print()

# Use the hsplit() method to split the 2-D array into three 2-D arrays along rows
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.hsplit(arr, 3)
print(newarr)
print()

# NumPy Searching Arrays
arr = np.array([1, 2, 3, 4, 5, 4, 4])
x = np.where(arr == 4)
print(x)
print()

# Find the indexes where the values are even
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 4, 4, 16, 17, 18])
x = np.where(arr%2 == 0)
print(x)
print()

# Find the indexes where the values are odd
x = np.where(arr%2 == 1)
print(x)
print()

# Search Sorted
arr = np.array([6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 4, 4, 16, 17, 18])
x = np.searchsorted(arr, 7)
print(x)
print()

# Search From the Right Side
x = np.searchsorted(arr, 17, side='right')
print(x)
print()

# Multiple Values
x = np.searchsorted(arr, [2, 4, 6, 17])
print(x)
print()

# NumPy Sorting Arrays
# Sorting Arrays
print(np.sort(arr))
print()
print(np.sort_complex(arr))
print()

# Sort a boolean array
arr1 = np.array([True, False, True])
print(np.sort(arr1))
print()

# Sorting a 2-D Array
arr = np.array([[13, 2, 6], [9, 7, 10]])
print(np.sort(arr))
print()

# NumPy Filter Array
# Filtering Arrays
arr = np.array([41, 42, 43, 44, 41, 42, 43, 44])
x = [True, False, True, False, True, False, False, True]
newarr = arr[x]
print(newarr)
print()

# Creating the Filter Array
arr0 = np.array([6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 4, 4, 16, 17, 18])
# Create an empty list
filter_arr = []

# go through each element in arr
for element in arr0:
  # if the element is higher than 42, set the value to True, otherwise False:
  if element > 5:
    filter_arr.append(True)
  else:
    filter_arr.append(False)

newarr = arr0[filter_arr]

print(filter_arr)
print(newarr)
print()

# Create a filter array that will return only even elements from the original array
# Create an empty list
filter_arr = []

# go through each element in arr
for element in arr0:
  # if the element is completely divisble by 2, set the value to True, otherwise False
  if element % 2 == 0:
    filter_arr.append(True)
  else:
    filter_arr.append(False)

newarr = arr0[filter_arr]

print(filter_arr)
print(newarr)
print()

# Creating Filter Directly From Array
filter_arr = arr0 > 4

newarr = arr0[filter_arr]

print(filter_arr)
print(newarr)
print()

# Create a filter array that will return only even elements from the original array
filter_arr = arr0 % 2 == 0
newarr = arr0[filter_arr]
print(filter_arr)
print(newarr)
print()

# Random Numbers in NumPy
x = np.random.randint(100)
print(x)
print()

# Generate Random Float
x = np.random.rand()
print(x)
print()

# Generate Random Array
x=np.random.randint(100, size=(15))
print(x)
print()

x = np.random.randint(100, size=(3, 5))
print(x)
print()

# Floats
x = np.random.rand(15)
print(x)
print()

x = np.random.rand(3, 5)
print(x)
print()

# Generate Random Number From Array
x = np.random.choice([3, 5, 7, 9])
print(x)
print()

x = np.random.choice([3, 5, 7, 9], size=(3, 5))
print(x)
print()

# Random Data Distribution
# Random Distribution
x = np.random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(100))
print(x)
print()

x = np.random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(3, 5))
print(x)
print()

# Random Permutations
# Shuffling Arrays
np.random.shuffle(arr0)
print(arr0)
print()

# Generating Permutation of Arrays
print(np.random.permutation(arr0))

# Seaborn
# Plotting a Displot
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot([0, 1, 2, 3, 4, 5])
#plt.show()

sns.distplot(arr0)
#plt.show()

# Plotting a Distplot Without the Histogram
sns.distplot([0, 1, 2, 3, 4, 5], hist=False)
#plt.show()

sns.distplot(arr0, hist=False)
#plt.show()

# Normal (Gaussian) Distribution
x = np.random.normal(size=(2, 3))
print(x)
print()

x = np.random.normal(loc=1, scale=2, size=(2, 3))
print(x)
print()

# Visualization of Normal Distribution
sns.distplot(np.random.normal(size=1000), hist=False)
#plt.show()

# Binomial Distribution
x = np.random.binomial(n=10, p=0.5, size=10)
print(x)
print()

# Visualization of Binomial Distribution
sns.distplot(np.random.binomial(n=10, p=0.5, size=1000), hist=True, kde=False)

# plt.show()

# Difference Between Normal and Binomial Distribution
sns.distplot(np.random.normal(loc=50, scale=5, size=1000), hist=False, label='normal')
sns.distplot(np.random.binomial(n=100, p=0.5, size=1000), hist=False, label='binomial')

#plt.show()

# Poisson Distribution
x = np.random.poisson(lam=2, size=10)

print(x)
print()

# Visualization of Poisson Distribution
sns.distplot(np.random.poisson(lam=10, size=1000), kde=False)

#plt.show()
print()

# Difference Between Normal and Poisson Distribution
sns.distplot(np.random.normal(loc=50, scale=7, size=1000), hist=False, label='normal')
sns.distplot(np.random.poisson(lam=50, size=1000), hist=False, label='poisson')

#plt.show()
print()

# Difference Between Poisson and Binomial Distribution
sns.distplot(np.random.binomial(n=1000, p=0.01, size=1000), hist=False, label='binomial')
sns.distplot(np.random.poisson(lam=10, size=1000), hist=False, label='poisson')

#plt.show()

# Uniform Distribution
x = np.random.uniform(size=(2, 4))
print(x)
print()

# Visualization of Uniform Distribution
import matplotlib.pyplot as plt1
import seaborn as sns1
sns1.distplot(np.random.uniform(size=1000), hist=False)
#plt1.show()

# Logistic Distribution
x = np.random.logistic(loc=1, scale=2, size=(2,15))
print(x)
print()

# Visualization of Logistic Distribution
sns.distplot(np.random.logistic(size=1000), hist=False)
#plt.show()

# Difference Between Logistic and Normal Distribution
sns.distplot(np.random.normal(scale=2, size=1000), hist=False, label='normal')
sns.distplot(np.random.logistic(size=1000), hist=False, label='logistic')

# plt.show()

# Multinomial Distribution
x = np.random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
print(x)
print()

# Exponential Distribution
x = np.random.exponential(scale=2, size=(2, 3))
print(x)
print()

# Visualization of Exponential Distribution
sns.distplot(np.random.exponential(size=1000), hist=False)
#plt.show()

sns.distplot(np.random.exponential(size=1000), hist=True)
# plt.show()

# Chi Square Distribution
x = np.random.chisquare(df=2, size=(2, 3))
print(x)

x = np.random.chisquare(df=3, size=(3, 4))
print(x)

# Visualization of Chi Square Distribution
sns.distplot(np.random.chisquare(df=1, size=1000), hist=False)
#plt.show()

sns.distplot(np.random.chisquare(df=1, size=1000), hist=True)
#plt.show()

sns.distplot(np.random.chisquare(df=1, size=1000))
#plt.show()
print()

# Rayleigh Distribution
x = np.random.rayleigh(scale=2, size=(2,3))
print(x)
print()

# Visualization of Rayleigh Distribution
sns.distplot(np.random.rayleigh(size=1000), hist=False)
#plt.show()
print()

# Pareto Distribution
x = np.random.pareto(a=2, size=(2, 3))
print(x)
print()

# Visualization of Pareto Distribution
sns.distplot(np.random.pareto(a=2, size=1000), kde=False)
# plt.show()
print()

# Zipf Distribution
x = np.random.zipf(a=2, size=(2, 3))
print(x)
print()

# Visualization of Zipf Distribution
x = np.random.zipf(a=2, size=1000)
sns.distplot(x[x<10], kde=False)
#plt.show()
print()

# NumPy ufuncs
# How To Create Your Own ufunc
def myadd(x, y):
  return x+y

myadd = np.frompyfunc(myadd, 2, 1)
print(myadd([1,2,3,4],[5,6,7,8]))
print()

# Check if a Function is a ufunc
print(type(np.add))
print()
print(type(np.concatenate))
print()

# Use an if statement to check if the function is a ufunc or not
if type(np.add) == np.ufunc:
  print('add is ufunc')
else:
  print('add is not ufunc')

# Simple Arithmetic
# add
arr1 = np.array([10,20,30,45,50,60])
arr2 = np.array([8,22,33,44,55,66])

newarr = np.subtract(arr1, arr2)
print(newarr)
print()

# Multiplication
arr1 = np.array([10,20,30,40,50,60])
arr2 = np.array([8,22,33,44,55,66])
newarr = np.multiply(arr1, arr2)
print(newarr)
print()

# Power
arr1 = np.array([-10,20,-30,40,-50,60])
arr2 = np.array([8,22,33,44,55,66])
newarr = np.power(arr1, arr2)
print(newarr)
print()


# Remainder
newarr = np.mod(arr1, arr2)
print(newarr)
print()
newarr = np.remainder(arr1, arr2)
print(newarr)
print()


# Quotient and Mod
newarr = np.divmod(arr1, arr2)
print(newarr)
print()


# Absolute Values
newarr = np.absolute(arr)
print(newarr)
print()

# Rounding Decimals
# Truncation
arr = np.trunc([-3.1666, 3.6667])
print(arr)
print()

arr = np.fix([-3.1666, 3.6667])
print(arr)
print()

# Rounding
arr = np.around(3.1666, 2)
print(arr)
print()

# Floor
arr = np.floor([-8.1666, 1.6667])
print(arr)
print()

# Ceil
arr = np.ceil([-4.14567, 2.9876])
print(arr)
print()

# NumPy Logs
# Log at Base 2
arr = np.arange(1, 10)
print(np.log2(arr))
print()

# Log at Base 10
arr = np.arange(1, 10)
print(np.log10(arr))
print()

# Natural Log, or Log at Base e
arr = np.arange(1, 10)
print(np.log(arr))
print()

# Log at Any Base
from math import log

nplog = np.frompyfunc(log, 2, 1)
print(nplog(100, 15))
print()

# NumPy Summations
# Summations
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
newarr = np.add(arr1, arr2)
print(newarr)
print()

arr1 = np.array([7,8,9])
arr2 = np.array([10,11,12])
newarr = np.sum([arr1, arr2])
print(newarr)
print()

# Summation Over an Axis
arr1 = np.array([7,8,9])
arr2 = np.array([10,11,12])
newarr = np.sum([arr1, arr2], axis=1)
print(newarr)
print()

# Cummulative Sum
newarr = np.cumsum(arr2)
print(newarr)
print()

# NumPy Products
# Products
arr = np.array([9,8,7,6])
x = np.prod(arr)
print(x)
print()

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])
x = np.prod([arr1, arr2])
print(x)
print()

# Product Over an Axis
arr1 = np.array([10, 21, 32, 43])
arr2 = np.array([54, 65, 76, 87])
newarr = np.prod([arr1, arr2], axis=1)
print(newarr)
print()

# Cummulative Product
arr = np.array([5, 6, 7, 8])
newarr = np.cumprod(arr)
print(newarr)
print()

# NumPy Differences
# Differences
arr = np.array([10,20,25,5])
newarr = np.diff(arr)
print(newarr)
print()

arr = np.array([11,13,15,17])
newarr = np.diff(arr, n=2)
print(newarr)
print()

# NumPy LCM Lowest Common Multiple
# Finding LCM (Lowest Common Multiple)
num1 = 4
num2 = 6
x = np.lcm(num1,num2)
print(x)
print()

# Finding LCM in Arrays
arr = np.array([3,5,7])
x = np.lcm.reduce(arr)
print(x)
print()


arr = np.arange(2, 22)
x = np.lcm.reduce(arr)
print(x)
print()

# NumPy GCD Greatest Common Denominator
# Finding GCD (Greatest Common Denominator)
num1 = 6
num2 = 9
x = np.gcd(num1, num2)
print(x)
print()

# Finding GCD in Arrays
arr = np.array([22,21,5,94,71])
x = np.gcd.reduce(arr)
print(x)
print()

# NumPy Trigonometric Functions
# Trigonometric Functions
x = np.sin(np.pi/2)
print(x)
print()


arr = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])
x = np.sin(arr)
print(x)
print()


# Convert Degrees Into Radians
arr = np.array([90, 180, 270, 360])
x = np.deg2rad(arr)
print(x)  
print()


# Radians to Degrees
arr = np.array([np.pi/2, np.pi, 1.5*np.pi, 2*np.pi])
x = np.rad2deg(arr)
print(x)
print()

# Finding Angles
x = np.arcsin(1.0)
print(x)
print()

# Angles of Each Value in Arrays
arr = np.array([1, -1, 0.1])
x = np.arcsin(arr)
print(x)
print()

# Hypotenues
base = 5
prep = 6
x1 = np.hypot(base, prep)
print(x1)
print()

# NumPy Hyperbolic Functions
x = np.sin(np.pi/2)
print(x)
print()

arr = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])
x = np.cosh(arr)
print(x)
print()

# NumPy Set Operations
# Create Sets in NumPy
arr = np.array([1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1])
x = np.unique(arr)
print(x)
print()

# Finding Union
arr1 = np.array([9,8,7,6,5,4,3,2,1,0])
arr2 = np.array([9,18,7,16,5,14,3,12,1,10])
newarr = np.union1d(arr1, arr2)
print(newarr)
print()

# Finding Intersection
arr1 = np.array([4,3,2,1])
arr2 = np.array([1,3,5,7])
newarr = np.intersect1d(arr1, arr2, assume_unique=True)
print(newarr)
print()

# Finding Difference
newarr = np.setdiff1d(arr1, arr2)
print(newarr)
print()

# Finding Symmetric Difference
newarr = np.setxor1d(arr1, arr2)
print(newarr)
print()
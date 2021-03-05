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
plt.show()

sns.distplot(arr0)
plt.show()

# Plotting a Distplot Without the Histogram
sns.distplot([0, 1, 2, 3, 4, 5], hist=False)
plt.show()

sns.distplot(arr0, hist=False)
plt.show()

# Normal (Gaussian) Distribution
x = np.random.normal(size=(2, 3))
print(x)
print()

x = np.random.normal(loc=1, scale=2, size=(2, 3))
print(x)
print()

# Visualization of Normal Distribution
sns.distplot(np.random.normal(size=1000), hist=False)
plt.show()

# Binomial Distribution
x = random.Binomial(n=10, p=0.5, sizwe=10)
print(x)
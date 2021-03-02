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
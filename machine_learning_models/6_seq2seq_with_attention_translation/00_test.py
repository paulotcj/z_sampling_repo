

#--------------------------------------------------
#-------
# 1 dim
list_1dim_1 = [ 1, 1, 1 ]
list_1dim_2 = [ 3, 3, 3 ]
list_1dim_3 = [ 5, 5, 5 ]

list_1dim_4 = [ 7, 7, 7 ]
list_1dim_5 = [ 11, 11, 11 ]
list_1dim_6 = [ 13, 13, 13 ]

list_1dim_7 = [ 17, 17, 17 ]
list_1dim_8 = [ 19, 19, 19 ]
list_1dim_9 = [ 23, 23, 23 ]
#-------
# 2 dim
list_2dim_1 = []
list_2dim_1.append(list_1dim_1)
list_2dim_1.append(list_1dim_2)
list_2dim_1.append(list_1dim_3)

list_2dim_2 = []
list_2dim_2.append(list_1dim_4)
list_2dim_2.append(list_1dim_5)
list_2dim_2.append(list_1dim_6)

list_2dim_3 = []
list_2dim_3.append(list_1dim_7)
list_2dim_3.append(list_1dim_8)
list_2dim_3.append(list_1dim_9)
#-------
# 3 dim
list_3dim_1 = []
list_3dim_1.append(list_2dim_1)
list_3dim_1.append(list_2dim_2)
list_3dim_1.append(list_2dim_3)

print('------------')
print(f'\n\list_3dim_1: {list_3dim_1}')
#--------------------------------------------------
#-------
# 1 dim
list_1dim_10 = [ 29, 29, 29 ]
list_1dim_11 = [ 31, 31, 31 ]
list_1dim_12 = [ 37, 37, 37 ]

list_1dim_13 = [ 41, 41, 41 ]
list_1dim_14 = [ 43, 43, 43 ]
list_1dim_15 = [ 47, 47, 47 ]

list_1dim_16 = [ 53, 53, 53 ]
list_1dim_17 = [ 59, 59, 59 ]
list_1dim_18 = [ 61, 61, 61 ]
#-------
# 2 dim
list_2dim_4 = []
list_2dim_4.append(list_1dim_10)
# list_2dim_4.append(list_1dim_11)
# list_2dim_4.append(list_1dim_12)

list_2dim_5 = []
list_2dim_5.append(list_1dim_13)
# list_2dim_5.append(list_1dim_14)
# list_2dim_5.append(list_1dim_15)

list_2dim_6 = []
list_2dim_6.append(list_1dim_16)
# list_2dim_6.append(list_1dim_17)
# list_2dim_6.append(list_1dim_18)

#-------
# 3 dim
list_3dim_2 = []
list_3dim_2.append(list_2dim_4)
list_3dim_2.append(list_2dim_5)
list_3dim_2.append(list_2dim_6)


print('------------')
print(f'\n\nlist_3dim_2: {list_3dim_2}')




result = list_3dim_1 + list_3dim_2

print('------------')
print(f'\n\nresult: {result}')


import torch
import numpy as np

# conver list to np.array
list_3dim_1 = np.array(list_3dim_1)
list_3dim_2 = np.array(list_3dim_2)

# convert np.array to torch tensor
list_3dim_1 = torch.from_numpy(list_3dim_1)
list_3dim_2 = torch.from_numpy(list_3dim_2)

print('------------')
print(f'list_3dim_1 type: {type(list_3dim_1)}')
print(f'list_3dim_1 shape: {list_3dim_1.shape}')
print(f'list_3dim_1: \n{list_3dim_1}')
print('------------')
print(f'list_3dim_2 type: {type(list_3dim_2)}')
print(f'list_3dim_2 shape: {list_3dim_2.shape}')
print(f'list_3dim_2: \n{list_3dim_2}')


result = list_3dim_1 + list_3dim_2

print('------------')
print(f'\n\nresult: \n  {result}')



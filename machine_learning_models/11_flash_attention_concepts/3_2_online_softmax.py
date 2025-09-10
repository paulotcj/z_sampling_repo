import torch

# input_vec = [
#     [ 0.11  ,  0.22  , 0.33  , 0.44  ],
#     [ 0.55  ,  0.77  , 0.22  , 0.11  ],
#     [ 0.001 ,  0.002 , 0.999 , 0.003 ]
# ]

input_vec = [
    [ 0.1  ,  0.2  , 0.5  , 0.4  ],
]

input_vec = torch.tensor(input_vec)


print(f'input_vec shape: {input_vec.shape}')
print(f'input_vec:\n{input_vec}')


print('----------------------------------------------')

# torch softmax as a reference
expected_softmax = torch.softmax(input_vec, dim=1)

print('\n\n')
print(f'expected_softmax:\n{expected_softmax}')

print('----------------------------------------------')
print('\n\n')

online_softmax = torch.zeros_like(input_vec)
print(f'online_softmax zeros:\n{online_softmax}')

print('----------------------------------------------')
print('\n\n')

'''
The core idea here is to find the 'row max value' and the denominator (normalizer_term) in a single
pass. 
In the  naive implementations of softmax, it might require multiple passes.

And then we will need another pass to apply the safe softmax and the denominator (normalizer_term)

Example: [1, 2, 3, 4] -> 
  exp(1) = 2.718 
    exp(1) / 2.718

  2.718 + exp(2) = 2.718 + 7.389 = 10.107
    exp(1) / 10.107 = 0.269
    exp(2) / 10.107 = 0.731

  10.107 + exp(3) = 10.107 + 20.086 = 30.193
    exp(1) / 30.193 = 0.090
    exp(2) / 30.193 = 0.245
    exp(3) / 30.193 = 0.665

  30.193 + exp(4) = 30.193 + 54.598 = 84.791
    exp(1) / 84.791 = 0.032
    exp(2) / 84.791 = 0.087
    exp(3) / 84.791 = 0.237
    exp(4) / 84.791 = 0.644

-----------
using our code
we have the tensor: [ 0.1  ,  0.2  , 0.5  , 0.4  ]

start row_max -> float('-inf') , normalizer_term = 0.0

  for 0.1 -----------------------------------------------
    row_max = 0.1  |  old_row_max = -inf. | col_v = 0.1 | normalizer_term = 0
      normalizer_term = normalizer_term * torch.exp(old_row_max - row_max) + torch.exp(col_v - row_max)
      normalizer_term = 0 * torch.exp(-inf - 0.1) + torch.exp(0.1 - 0.1)
      normalizer_term = 0 * torch.exp(-inf) + torch.exp(0) 
      normalizer_term = 0 * 0.0 + 1.0
      normalizer_term = 0.0 + 1.0
      normalizer_term = 1.0

  for 0.2 -----------------------------------------------
    row_max = 0.2  |  old_row_max = 0.1 | col_v = 0.2 | normalizer_term = 1
      normalizer_term = normalizer_term * torch.exp(old_row_max - row_max) + torch.exp(col_v - row_max)
      normalizer_term = 1 * torch.exp(0.1 - 0.2) + torch.exp(0.2 - 0.2)
      normalizer_term = 1 * torch.exp(-0.1) + torch.exp(0)
      normalizer_term = 1 * 0.9048 + 1.0
      normalizer_term = 0.9048 + 1.0
      normalizer_term = 1.9048

  for 0.5 -----------------------------------------------
    row_max = 0.5  |  old_row_max = 0.2 | col_v = 0.5 | normalizer_term = 1.9048
      normalizer_term = normalizer_term * torch.exp(old_row_max - row_max) + torch.exp(col_v - row_max)
      normalizer_term = 1.9048 * torch.exp(0.2 - 0.5) + torch.exp(0.5 - 0.5)
      normalizer_term = 1.9048 * torch.exp(-0.3) + torch.exp(0)
      normalizer_term = 1.9048 * 0.7408 + 1.0
      normalizer_term = 1.4111 + 1.0
      normalizer_term = 2.4111

  for 0.4 -----------------------------------------------
    row_max = 0.5  |  old_row_max = 0.5 | col_v = 0.4 | normalizer_term = 2.4111
      normalizer_term = normalizer_term * torch.exp(old_row_max - row_max) + torch.exp(col_v - row_max)
      normalizer_term = 2.4111 * torch.exp(0.5 - 0.5) + torch.exp(0.4 - 0.5)
      normalizer_term = 2.4111 * torch.exp(0.0) + torch.exp(-0.1)
      normalizer_term = 2.4111 * 1.0 + 0.9048
      normalizer_term = 2.4111 + 0.9048
      normalizer_term = 3.3160
'''

# paper: Online normalizer calculation for softmax - https://arxiv.org/pdf/1805.02867
#-----------------------------------
for row_k, row_v in enumerate(input_vec):
    row_max = float('-inf')
    normalizer_term = 0.0
    print(f'row {row_k} -----------------------------------------------------')

    #-----------------------------------
    for col_k, col_v in enumerate(row_v):
        print(f'    col {col_k} ---------')

        #---- 
        old_row_max = row_max
        row_max = max(old_row_max, col_v)
        if old_row_max != row_max:
            print(f'        new max discovered: {row_max:.4f}')        
        #----

        normalizer_term = normalizer_term * torch.exp(old_row_max - row_max) + torch.exp(col_v - row_max)

        print(f'        current row max: {row_max:.4f}, denominator: {normalizer_term:.4f}')
    #-----------------------------------

    # old code
    # #------
    # # this section is pretty standard, you can compare with regular/safe softmax
    # #   the only different thing is how 'normalizer_term' was calculated and row_max was found
    # input_safe = input_vec[row_k] - row_max
    # temp = torch.exp( input_safe ) / normalizer_term

    # online_softmax[row_k] = temp
    # #------

    # following the code strictly similar to the original paper: Online normalizer calculation for softmax
    #   https://arxiv.org/pdf/1805.02867 , page 3
    #-----------------------------------
    for col_k, col_v in enumerate(row_v):
        input_safe = col_v - row_max # input safe
        temp = torch.exp(input_safe) / normalizer_term
        online_softmax[row_k, col_k] = temp
    #-----------------------------------


#-----------------------------------

print('----------------------------------------------')
print('\n\n')
print(f'online_softmax:\n{online_softmax}')
print(f'expected_softmax:\n{expected_softmax}')
print('----------------------------------------------')
print('\n\n')

print(f'torch.allclose(online_softmax, expected_softmax): {torch.allclose(online_softmax, expected_softmax)}')
with open('input.txt', 'r') as f:
    text = f.read()

print(f'text len:{len(text)}')
data = text[:1000] # first 1,000 characters
print(data[:100])



print('-------------------------------------------------------------------------')
print('\n\n')
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(data)
print(f'tokens:\n{tokens}')
print(f'tokens len: {len(tokens)}')
'''
this is the data we want to use to train
tokens:
[5962, 22307, 25, 198, 8421, 356, 5120, 597, 2252, 11, 3285, 502, 2740, 13, 198, 198, 3237, 25, 198, 
5248, 461, 11, 2740, 13, 198, 198, 5962, 22307, 25, 198, 1639, 389, 477, 12939, 2138, 284, 4656, 621, 
284, 1145, 680, 30, 198, 198, 3237, 25, 198, 4965, 5634, 13, 12939, 13, 198, 198, 5962, 22307, 25, 
198, 5962, 11, 345, 760, 327, 1872, 385, 1526, 28599, 318, 4039, 4472, 284, 262, 661, 13, 198, 198, 
3237, 25, 198, 1135, 760, 470, 11, 356, 760, 470, 13, 198, 198, 5962, 22307, 25, 198, 5756, 514, 1494, 
683, 11, 290, 356, 1183, 423, 11676, 379, 674, 898, 2756, 13, 198, 3792, 470, 257, 15593, 30, 198, 
198, 3237, 25, 198, 2949, 517, 3375, 319, 470, 26, 1309, 340, 307, 1760, 25, 1497, 11, 1497, 0, 198, 
198, 12211, 22307, 25, 198, 3198, 1573, 11, 922, 4290, 13, 198, 198, 5962, 22307, 25, 198, 1135, 389, 
17830, 3595, 4290, 11, 262, 1458, 1173, 1547, 922, 13, 198, 2061, 4934, 969, 5036, 896, 319, 561, 
26958, 514, 25, 611, 484, 198, 19188, 7800, 514, 475, 262, 48713, 414, 11, 981, 340, 547, 198, 1929, 
4316, 462, 11, 356, 1244, 4724, 484, 22598, 514, 31533, 306, 26, 198, 4360, 484, 892, 356, 389, 1165, 
13674, 25, 262, 10904, 1108, 326, 198, 2001, 42267, 514, 11, 262, 2134, 286, 674, 24672, 11, 318, 355, 
281, 198, 24807, 284, 1948, 786, 511, 20038, 26, 674, 198, 82, 13712, 590, 318, 257, 4461, 284, 606, 
3914, 514, 15827, 428, 351, 198, 454, 279, 7938, 11, 304, 260, 356, 1716, 374, 1124, 25, 329, 262, 
11858, 760, 314, 198, 47350, 428, 287, 16460, 329, 8509, 11, 407, 287, 24613, 329, 15827, 13, 628]

tokens len: 285
'''


print('-------------------------------------------------------------------------')
print('\n\n')
import torch

buf = torch.tensor(tokens[:24])
x = buf.view(4, 6)
print(f'x:\n{x}')
print(f'x shape: {x.shape}')

'''
not lets imagine we try to shape it this way for training

x:
tensor([[ 5962, 22307,    25,   198,  8421,   356],
        [ 5120,   597,  2252,    11,  3285,   502],
        [ 2740,    13,   198,   198,  3237,    25],
        [  198,  5248,   461,    11,  2740,    13]])
x shape: torch.Size([4, 6])

The problem here is that with X only we have an issue of how to validate if
the next token prediction is correct, as we have to provide a block of 6 tokens
for prediction and then check against a block of 6 tokens for validation
'''

print('-------------------------------------------------------------------------')
print('\n\n')


buf = torch.tensor(tokens[:24 + 1]) # + 1 as this is intended to accomodate the Y for validation, see det below
x = buf[:-1].view(4, 6) # get everything minus the last element, then shape it into a view of 4 rows by 6 cols
y = buf[1:].view(4, 6) # get everything minus the fist element, then shape it into a view of 4 rows by 6 cols
print(f'x:\n{x}')
print(f'y:\n{y}')

'''
This approach can be considered a sliding window, where for X we get all elements minus the last one, as
the last one will be always predicted.
Y will get all the elements minus until the end, minus the first, as the first is always given and 
never predicted.
This is a little bit confusing, so let's take the example below:

x:
tensor([[ 5962, 22307,    25,   198,  8421,   356],
        [ 5120,   597,  2252,    11,  3285,   502],
        [ 2740,    13,   198,   198,  3237,    25],
        [  198,  5248,   461,    11,  2740,    13]])
y:
tensor([[22307,    25,   198,  8421,   356,  5120],
        [  597,  2252,    11,  3285,   502,  2740],
        [   13,   198,   198,  3237,    25,   198],
        [ 5248,   461,    11,  2740,    13,   198]])

We have blocks of 6 tokens, we feed the first block from 
X -> [ 5962, 22307,    25,   198,  8421,   356]
and we expect to get 
Y ->        [22307,    25,   198,  8421,   356,  5120]
'''

print('-------------------------------------------------------------------------')
print('\n\n')

from transformers import GPT2LMHeadModel
model_hf = GPT2LMHeadModel.from_pretrained("gpt2") # 124M
sd_hf = model_hf.state_dict()



print(f'sd_hf["lm_head.weight"].shape: {sd_hf["lm_head.weight"].shape}')
print(f'sd_hf["transformer.wte.weight"].shape: {sd_hf["transformer.wte.weight"].shape}')

print('\n\n')
print(f'(sd_hf["lm_head.weight"] == sd_hf["transformer.wte.weight"]).all(): {(sd_hf["lm_head.weight"] == sd_hf["transformer.wte.weight"]).all()}')

print('\n\n')
print(f'sd_hf["lm_head.weight"].data_ptr(): {sd_hf["lm_head.weight"].data_ptr()}')
print(f'sd_hf["transformer.wte.weight"].data_ptr(): {sd_hf["transformer.wte.weight"].data_ptr()}')


'''
Let's investigate these 2 tensors

sd_hf["lm_head.weight"].shape         : torch.Size([50257, 768])
sd_hf["transformer.wte.weight"].shape : torch.Size([50257, 768])

same shape

comparing element wise we see they are identical:
(sd_hf["lm_head.weight"] == sd_hf["transformer.wte.weight"]).all(): True


comparing the pointer we see it's the same tensor:
sd_hf["lm_head.weight"].data_ptr()        : 13714325504
sd_hf["transformer.wte.weight"].data_ptr(): 13714325504

These 2 are at the beginning and at the end of the model. The very first item is:
transformer.wte.weight torch.Size([50257, 768])   - word token embedding
and 149 items below, the last item:
lm_head.weight torch.Size([50257, 768])           - language model head

From the original paper 'attention is all you need' they disclose they are using the same
  tensor 'wte' in both ends as the expectation is that if 2 tokens are semantically similar
  their embeddings should also be similar, and therefore that's why they share the same  word
  token embedding. This was originally discussed in a paper from 2017 'Using the Output Embedding
  to Improve Language Models'
'''


from transformers import GPT2LMHeadModel
print('-------------------------------------------------------------------------')
print('\n\n')

model_hf = GPT2LMHeadModel.from_pretrained("gpt2") # 124M
sd_hf = model_hf.state_dict() # raw tensors

print('-------------------------------------------------------------------------')
print('\n\n')
for k, v in sd_hf.items():
    key_output = f'key: {k}                                  '[0:41]
    print(key_output, end = '')
    print(f'value: {v.shape}\n', end = '')

    '''
    and we will see things as:
       key: transformer.wte.weight   value: torch.Size([50257, 768]) -> weights for token embedding
          vocab/tokens size of 50257 and dim 768
    '''

print('-------------------------------------------------------------------------')
print('\n\n')

# weights for positional embedding
# create a view of the tensor, -1 flattens the tensor into a one-dimensional array, and then
#   slice it to the first 20 items
temp = sd_hf["transformer.wpe.weight"].view(-1)[:20]
print(temp)


print('-------------------------------------------------------------------------')
print('\n\n')
import matplotlib.pyplot as plt

'''
notice that for this item the details are: key: 
    transformer.wpe.weight              value: torch.Size([1024, 768])
meaning we have 1024 positions for positional encoding weights 
'''
plt.imshow(sd_hf["transformer.wpe.weight"], cmap="gray")
plt.show()


print('-------------------------------------------------------------------------')
print('\n\n')

# let's sample 3 columns
plt.plot(sd_hf["transformer.wpe.weight"][:, 150])
plt.plot(sd_hf["transformer.wpe.weight"][:, 200])
plt.plot(sd_hf["transformer.wpe.weight"][:, 250])
plt.show()
'''
it's hard to make sense of why they are this way, but it's interesting to note they have
   a wavy pattern
'''




print('-------------------------------------------------------------------------')
print('\n\n')
plt.imshow(sd_hf["transformer.h.1.attn.c_attn.weight"][:300,:300], cmap="gray")
plt.show()


print('-------------------------------------------------------------------------')
print('\n\n')

from transformers import pipeline, set_seed
set_seed(42)

# if we were to use the big one that would be 'gpt2-xl'
generator = pipeline(task = 'text-generation', model = 'gpt2') # if we were to use the big one
 
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
result = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
print('-------------------------------------------------------------------------')
print('\n\n')

for i in result:
    print(i)

with open('input.txt', 'r') as f:
    text = f.read()

data = text[:1000] # first 1,000 characters
print(data[:100])


print('-------------------------------------------------------------------------')
print('\n\n')
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(data)
# print(tokens[:24])


print('-------------------------------------------------------------------------')
print('\n\n')
import torch
buf = torch.tensor(tokens[:24 + 1])

print(f'buf:\n{buf}')
print(f'buf shape: {buf.shape}')
print('-------------------------------------------------------------------------')
print('\n\n')
print(f'buf[:-1]:\n{buf[:-1]}') # get everything from begining minus the last element
print(f'buf[1:]:\n{buf[1:]}') # get everything from positin 1 to the last element


print('-------------------------------------------------------------------------')
print('\n\n')
x = buf[:-1].view(4, 6) # same as above bur arrange the elemenets in 4 rows by 6 columns
y = buf[1:].view(4, 6)

print(f'x = buf[:-1].view(4, 6):\n{x}')
print(f'y = buf[1:].view(4, 6):\n{y}')

print('-------------------------------------------------------------------------')

from transformers import GPT2LMHeadModel
model_hf = GPT2LMHeadModel.from_pretrained("gpt2") # 124M
sd_hf = model_hf.state_dict()

print(sd_hf["lm_head.weight"].shape)
print(sd_hf["transformer.wte.weight"].shape)


print('-------------------------------------------------------------------------')
print((sd_hf["lm_head.weight"] == sd_hf["transformer.wte.weight"]).all())


print('-------------------------------------------------------------------------')
print(sd_hf["lm_head.weight"].data_ptr())
print(sd_hf["transformer.wte.weight"].data_ptr())


print('-------------------------------------------------------------------------')
# standard deviation grows inside the residual stream
x = torch.zeros(768)

n = 100 # e.g. 100 layers
for i in range(n):
    x += n**-0.5 * torch.randn(768)

print(x.std())
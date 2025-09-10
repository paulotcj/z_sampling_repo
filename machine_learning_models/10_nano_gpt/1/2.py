
# let's instead sample manually
import torch
from torch.nn import functional as F

#-------------------------------------------------------------------------
def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda' 
        print('using cuda acceleration')
    elif torch.backends.mps.is_built():
        device = 'mps'
        print('using mps acceleration')
    else:
        device = 'cpu'
        print('using cpu')

    return device
#-------------------------------------------------------------------------


print('-------------------------------------------------------------------------')
print('\n\n')

device = get_device()

from transformers import GPT2LMHeadModel
import tiktoken

enc = tiktoken.get_encoding("gpt2") 

#-------------------------------------------------------------------------
str_to_encode = "Hello, I'm a language model,"

int_list_encoded = enc.encode(text = str_to_encode)

str_decoded = enc.decode(tokens=int_list_encoded)
print(f'str_to_encode    :{str_to_encode}')
print(f'int_list_encoded :{int_list_encoded}')
print(f'str_decoded      :{str_decoded}')

#-------------------------------------------------------------------------


model = GPT2LMHeadModel.from_pretrained( pretrained_model_name_or_path= "gpt2") # 124M
model.eval()
model.to(device)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
tokens = int_list_encoded
tokens = torch.tensor(data = tokens, dtype=torch.long) # (8,)
print(f'tokens:\n{tokens}')
tokens = tokens.unsqueeze(dim = 0).repeat(5, 1) # (5, 8)
print(f'tokens:\n{tokens}')
x = tokens.to(device)
print(f'x shape:{x.shape}')
print(f'x.size(1):{x.size(1)}')


#-------------------------------------------------------------------------
# x [5,8] - the gist here is that we have X with 5 sentences of length 8, and we will
#   loop about 30 times asking the model for a new token (which is not necessarily a word or
#   a single character)
seq_max_len = 30
while x.size(1) < seq_max_len: 
    with torch.no_grad():
        logits = model(x)[0] # forward the model to get the logits - (B, T, vocab_size)

        # take the logits at the last position - all rows, last column only, all chars
        logits = logits[:, -1, :] # (B, vocab_size)

        probs = F.softmax(input = logits, dim = -1)

        # do top-k sampling of 50 (huggingface pipeline default) - topk_probs here becomes (5, 50), topk_indices is (5, 50)
        top_k_probs, top_k_indices = torch.topk(input = probs, k = 50, dim=-1)


        # select a token from the top-k probabilities - note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(input = top_k_probs, num_samples = 1) # (B, 1)

        # gather the corresponding indices
        x_col = torch.gather(input = top_k_indices, dim = -1, index = ix) # (B, 1)

        x = torch.cat((x, x_col), dim=1)
#-------------------------------------------------------------------------


# print the generated text - x is 30 sentences of 30 tokens in length
for i in range(5):
    tokens = x[i, :30].tolist() # get sentence i_th, select tokens from 0 to 30
    decoded = enc.decode(tokens)
    print(f'> {decoded}')
    

#-------------------------------------------------------------------------
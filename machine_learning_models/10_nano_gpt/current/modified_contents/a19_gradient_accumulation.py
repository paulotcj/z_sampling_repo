'''
step   10 | loss: 7.028956 | lr 6.0000e-04 | norm: 1.8351 | dt: 3855.56ms | tok/sec: 135982.33
step   11 | loss: 6.740683 | lr 5.9917e-04 | norm: 1.5009 | dt: 3868.75ms | tok/sec: 135518.85
step   12 | loss: 6.528240 | lr 5.9668e-04 | norm: 1.1483 | dt: 3703.16ms | tok/sec: 141578.56
step   13 | loss: 6.376919 | lr 5.9254e-04 | norm: 1.0641 | dt: 3974.04ms | tok/sec: 131928.11
step   14 | loss: 6.339537 | lr 5.8679e-04 | norm: 2.5897 | dt: 3901.56ms | tok/sec: 134379.05
step   15 | loss: 6.243823 | lr 5.7945e-04 | norm: 1.0017 | dt: 3951.39ms | tok/sec: 132684.46
step   16 | loss: 6.212921 | lr 5.7057e-04 | norm: 0.7823 | dt: 4396.17ms | tok/sec: 119260.20
step   17 | loss: 6.210563 | lr 5.6021e-04 | norm: 1.1362 | dt: 3956.24ms | tok/sec: 132521.91
step   18 | loss: 6.156693 | lr 5.4843e-04 | norm: 0.9848 | dt: 3945.69ms | tok/sec: 132875.96
step   19 | loss: 6.148070 | lr 5.3531e-04 | norm: 1.5321 | dt: 3941.94ms | tok/sec: 133002.70
'''
import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


#-------------------------------------------------------------------------
'''
Causal: In the context of GPT-2, "causal" means that the attention mechanism is restricted to only 
consider previous tokens in the sequence. This is crucial for autoregressive models like GPT-2, which 
generate text one token at a time. The model should not have access to future tokens during training 
or inference.
'''
class CausalSelfAttention(nn.Module): # multi head attention
    #-------------------------------------------------------------------------
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch, instead of 3 linear layers for q,v,k we have
        #   just 1 with enough bandwith for the 3 of them
        #  concatenated attention - https://arxiv.org/pdf/1706.03762 - page 4 - Multi-Head Attention - right image
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # (768, 768 * 3) -> (768, 2304)

        # output projection
        #  concatenated projection - https://arxiv.org/pdf/1706.03762 - page 4 - Multi-Head Attention - right image
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
	
        self.c_proj.NANOGPT_SCALE_INIT = 1
	
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        mask = torch.tril( #triangular matrix
            torch.ones(config.block_size, config.block_size) # ones matrix of size (block_size, block_size) -> (1024, 1024)
        ).view(1, 1, config.block_size, config.block_size) # creates a view with dim (1,1,1024,1024) basically adds 2 more dummy dims

        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", mask)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd) B = 5, T = 8, C = 768

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(input = x) # [5, 8, 2304]

        # all [5, 8, 768]
        q, k, v = qkv.split(split_size = self.n_embd, dim=2) # split_size = 768, but the output is 2304, so 3 splits

        # the view splits the last dimension (768) and doesn't alter the first one, as their
        #   numbers were unchanged. But in general this is a simple reorganization as the number
        #   of elements didn't change.
        #   [5, 8, 12, 64] |||  from [5, 8, 768] -> view [5, 8, 12, 64] -> [5, 12, 8, 64]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) 


        #--------------------------------------------------
        # Replace this by flash attention - start

        # # multiply Q (query) by K (key), and traspose the second matrix K, then apply the scaling factor
        # # attention (materializes the large (T,T) matrix for all the queries and keys)
        # # att [5, 12, 8, 8] <- q [5, 12, 8, 64] ,  k.trans [5, 12, 64, 8]
        # att = (q @ k.transpose(-2, -1)) * ( 1.0 / math.sqrt(k.size(-1)) )

        # # apply mask - don't let it peek into the future tokens
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        # att = F.softmax(att, dim=-1)

        # # y [5, 12, 8, 64] , att [5, 12, 8, 8]  , v [5, 12, 8, 64]
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)


        # # .contiguous(): This ensures that the tensor's memory layout is contiguous.
        # # y [5, 8, 768]   |   y.trans.cont [5, 8, 12, 64] -> y.view [5, 8, 768] | B = 5, T = 8, C = 768
                

        # Replace this by flash attention - end
        #--------------------------------------------------
        '''
        The old code above was challenging for torch.compile to optimize. Flash attention is a 'kernel
        fusion' operation. Flash attention can be up to 7.6X faster than the old implementation
        '''

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
	
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection  - [5, 8, 768]
        y = self.c_proj(y)

        return y
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class MLP(nn.Module):
    #-------------------------------------------------------------------------
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd) # context fully connected
        self.gelu    = nn.GELU(approximate='tanh') # GELU activation will use the tanh approximation for its computation, which can be faster than the exact computation.
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd) # context projection
        self.c_proj.NANOGPT_SCALE_INIT = 1
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class Block(nn.Module):
    #-------------------------------------------------------------------------
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
@dataclass
class GPTConfig:
    # note you can define the values here in any order
    block_size : int = 1024 # max sequence length
    vocab_size : int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer    : int = 12 # number of layers
    n_head     : int = 12 # number of heads
    n_embd     : int = 768 # embedding dimension
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
    '''
    This implementation of GPT follows the same pattern of GPT2. When exploring the
      open weights from GPT2 128M model we would find something like this:
            transformer.wte.weight torch.Size([50257, 768])
            transformer.wpe.weight torch.Size([1024, 768])
            transformer.h.0.ln_1.weight torch.Size([768])
            transformer.h.0.ln_1.bias torch.Size([768])
            transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
            transformer.h.0.attn.c_attn.bias torch.Size([2304])
            transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
            transformer.h.0.attn.c_proj.bias torch.Size([768])
            transformer.h.0.ln_2.weight torch.Size([768])
            transformer.h.0.ln_2.bias torch.Size([768])
            transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
            transformer.h.0.mlp.c_fc.bias torch.Size([3072])
            transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
            transformer.h.0.mlp.c_proj.bias torch.Size([768])

            [...] LAYERS FROM 0 TO 11
            
            transformer.h.11.mlp.c_proj.weight torch.Size([3072, 768])
            transformer.h.11.mlp.c_proj.bias torch.Size([768])
            transformer.ln_f.weight torch.Size([768])
            transformer.ln_f.bias torch.Size([768])
            lm_head.weight torch.Size([50257, 768])

    -------------------

    First we should define some basic hyper parameters: 
        - 50257 is the token vocab size
        - 768 is the embedding dimension
        - 1024 is the block size

    -------------------

    As we can see GPT is composed of: 
        - transformer, which will be explained below
        - lm_head (language model head - LinearLayer) in_features = 50257 (vocab_size), out_features = 768 (embedding dimension), bias  = False
    
    -------------------

    The transformer is composed of:
        - wte (word token embeddings   - Embedding) with num_embeddings = 50257 (vocab_size) and embeddings_dim = 768 (embedding dimension)
        - wpe (word position embedding - Embedding) with num_embeddings = 1024 (block size) and embeddings_dim = 768 (embedding dimension)
        - h (hidden layers, from 0 to 11 to be explained below - ModuleList[Block])
        - ln_f (layer normalization final - LayerNorm) ln_f.weight normalized_shape = 768 (embedding dimension) ln_f.bias normalized_shape = 768 (embedding dimension)

    -------------------

    The h (hidden layers - ModuleList[Block]) is composed of 12 hidden layers of the following components:
        - ln_1 (layer normalization LayerNorm), ln_1.weight normalized_shape = 768, ln_1.bias normalized_shape = 768
        - attn (CausalSelfAttention - note that this is CAUSAL)
        - ln_2 (layer normalization LayerNorm), ln_2.weight normalized_shape = 768, ln_2.bias normalized_shape = 768
        - mlp (MLP Multi Layer Perceptron)

    -------------------

    The CausalSelfAttention
        - c_attn (attention  - linear layer) in_features = 768 (embedding dimension) out_features = 2304 (3 * 768 embedding dimension), bias = 2304 (3 * 768 embedding dimension) 
        - c_proj (projection - linear layer) in_features = 768 (embedding dimension), out_features = 768 (embedding dimension), bias = 768 (embedding dimension) 

    -------------------

    The MLP (Multi Layer Perceptron)
        - c_fc (fully connected - linear layer) in_features = 768 (embedding dimension), out_features = 3072 (4 * 768), bias = 3072 (4 * 768)
        - c_proj (projection    - linear layer) in_features = 3072 (4 * 768), out_features = 768 (embedding dimension), bias = 768 (embedding dimension)

    '''
class GPT(nn.Module):
    #-------------------------------------------------------------------------
    def __init__(self, gpt_config : GPTConfig):
        super().__init__()

        self.config = gpt_config

        # below we need to define the following:
        #   transformer and lm_head
        #
        # and transformer is composed of: 
        #   wte, wpe, h, ln_f
        #--------
        # transformer dict - define wte, wpe, h, ln_f

        # h - hidden layers. composed of a list of Block objects
        block_list = [ Block(gpt_config) 
                       for _ in range(gpt_config.n_layer) ]

        # the transformer dictionary 
        transf_dict = dict(
                wte  = nn.Embedding(gpt_config.vocab_size, gpt_config.n_embd),
                wpe  = nn.Embedding(gpt_config.block_size, gpt_config.n_embd),
                h    = nn.ModuleList(block_list),
                ln_f = nn.LayerNorm(gpt_config.n_embd) )
        #--------
        # define transformer
        self.transformer = nn.ModuleDict( transf_dict )
        #--------
        #define lm_head
        self.lm_head = nn.Linear(gpt_config.n_embd, gpt_config.vocab_size, bias=False)
        #--------

        '''
        An explanation is necessary for the weight sharing below: transformer.wte (token embedding)
        is used to convert input token IDs into dense vectors. lm_head is used to convert the final 
        hidden states back into token probabilities for the output. By sharing weights, the model 
        ensures that the same vector space is used for both input and output tokens. This symmetry 
        can help the model learn more effectively, as the same representation is used consistently 
        throughout the network. Plus, research and empirical results have shown that weight sharing 
        can lead to better performance in language models
        '''
	
        # weight sharing scheme - transformer token embeddings take the weights of the language model head weights
        self.transformer.wte.weight = self.lm_head.weight
	

        # init params
        self.apply(self._init_weights)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def _init_weights(self, module):
        # just trying to make sure things are initialized correctly. 
        #   Linear layers should be std = 0.02 and a mean of 0, except if they have a flag NANOGPT_SCALE_INIT
        #     this flag is reserved for accumulation residual layers in this case they should be 
        #     0.02 * 1 / sqrt(2 * self.config.n_layer) -> 0.02 * (2 * self.config.n_layer) ** -0.5
        #     this helps to control the growth of activations in the forward pass (for res nets)
        #     This will typically affect the C_PROJ (linear) of MLP and CausalSelfAttention. You can
        #     confirm these are the 2 last res nets of these components.
        # Note that the values around 0.02 are a mere simplification of the Xavier initialization.
        #   for many parameters as in d_model = 768, we would have 1/sqrt(768) = 0.03608439182
        #   or if we were using d_model as 1600, we would have 1/sqrt(1600) = 0.025

        if isinstance(module, nn.Linear):
            std = 0.02

            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # forward the token and posisition embeddings
        #  create a 1-dim tensor 
        pos = torch.arange(start = 0, end = T, dtype=torch.long, device=idx.device) # shape (T)

        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        
        x = tok_emb + pos_emb # residual connection / skip connection
        
        # forward the blocks of the transformer hidden layers
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)

        # linear layer
        logits = self.lm_head(x) # (B, T, vocab_size)


        loss = None
        if targets is not None:

            # for logits.view(-1, logits.size(-1)):
            #    -1 means guess the size, while we define the size of the last dim as being 5027
            #    since in one example logits is [4, 32, 50257], that means [ 4 * 32 , 5027 ] -> [ 128, 5027 ]
            # for targets.view(-1)
            #     we are amalgamating targets into a single dim, therefore [128], targets shape is [4, 32]

            loss = F.cross_entropy(input = logits.view(-1, logits.size(-1)) , target = targets.view(-1))


        return logits, loss
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    '''
    The decorator below (@classmethod) is used to define a method that is bound to the 
    class and not the instance of the class. This means that when you call a class method, 
    it receives the class as its first argument, which is conventionally named cls. This 
    is similar to how instance methods receive the instance as their first argument, 
    conventionally named self.
    They are often used for methods that need to access or modify  class-level attributes, 
    rather than instance-level attributes.
    For instance:
        class MyClass:
            class_attribute = 0
            ...
            @classmethod
            def increment_class_attribute(cls):
                cls.class_attribute += 1
    '''
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""

        #-------
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        #-------

        #-------
        # select pre config of the model type
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        print('-------')
        print('config_args.items()')
        for k,v in config_args.items():
            print(f"    k: {k} - v: {v}")
        print('-------')    
        #-------

        #
        # create a from-scratch initialized minGPT model
        gpt_config = GPTConfig(**config_args) # the values from config can be passed in any order

        print(f'config: {gpt_config}')

        #-------------------
        # create our own model and its state dictionary
        model = GPT(gpt_config)

        # we are going to set the values to state_dict, but since this is a reference and any updates here will
        #   reflect at model
        state_dict = model.state_dict() # from nn.Module        
        state_dict_keys = state_dict.keys()

     

        print('-------')  
        print('state_dict_keys')
        for i in state_dict_keys:
            print(f"    k: {i}")
        print('-------')


        # filter out '.attn.bias' from state_dict_keys
        state_dict_keys = [
            k 
            for k in state_dict_keys 
            if not k.endswith('.attn.bias')
        ] # discard this mask / buffer, not a param


        #-----------
        removed_items = sorted(
                list( 
                set( state_dict.keys() )
                .symmetric_difference( 
                    set(state_dict_keys) 
                ) 
            )
        )
        print('-------')  
        print('removed_items')
        for i in removed_items:
            print(f"    i: {i}")
        print('-------')
        #-----------
        #-------------------

        #-------------------
        # Let's grab the model from HuggingFace and then filter out what we don't want
        # init a huggingface/transformers model
        model_hugginface = GPT2LMHeadModel.from_pretrained(model_type)
        state_dict_huggingface = model_hugginface.state_dict()


        # copy while ensuring all of the parameters are aligned and match in names and shapes
        state_dict_keys_huggingface = state_dict_huggingface.keys()

        #-------
        # ignore ".attn.masked_bias" it's just a buffer, and ".attn.bias" is a mask (buffer)
        state_dict_keys_huggingface = [
            k 
            for k in state_dict_keys_huggingface 
            if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')
        ]
        #-------

        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        #-------------------

        #-------------------
        # copy phase
        # at this point the dict state from hugging face and our dict state should be identical
        assert len(state_dict_keys_huggingface) == len(state_dict_keys), f"mismatched keys: {len(state_dict_keys_huggingface)} != {len(state_dict_keys)}"


        for k in state_dict_keys_huggingface:

            if any( k.endswith(w) for w in transposed ): # let's transpose what needs to transposed
                # special treatment for the Conv1D weights we need to transpose
                #   e.g.: state_dict_huggingface[k].shape ---> torch.Size([768, 2304])
                #     state_dict_huggingface[k].shape[::-1] -> torch.Size([2304, 768])
                #     state_dict[k].shape -------------------> torch.Size([2304, 768])
                assert state_dict_huggingface[k].shape[::-1] == state_dict[k].shape

                with torch.no_grad():
                    transposed_component = state_dict_huggingface[k].t()
                    state_dict[k].copy_( transposed_component )
            else:
                # vanilla copy over the other parameters
                assert state_dict_huggingface[k].shape == state_dict[k].shape
                with torch.no_grad():
                    state_dict[k].copy_(state_dict_huggingface[k])

            #----
            

        #-------------------


        return model
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def configure_optimizers(self, weight_decay, learning_rate, device):

        #-----------------
        # start with all of the candidate parameters (that require grad)
        '''
        these are params like:
            transformer.wte.weight torch.Size([50304, 768])
            transformer.wpe.weight torch.Size([1024, 768])
            transformer.h.0.ln_1.weight torch.Size([768])
            transformer.h.0.ln_1.bias torch.Size([768])
            transformer.h.0.attn.c_attn.weight torch.Size([2304, 768])
        '''
        param_dict = { param_name: param 
                      for param_name, param in self.named_parameters()
                      if param.requires_grad 
        }
        
        # param_dict = {param_name: param 
        #               for param_name, param in param_dict.items()
        #               if param.requires_grad
        # }


        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [ param 
                        for param_name, param in param_dict.items() 
                        if param.dim() >= 2
        ]

        nodecay_params = [param 
                          for param_name, param in param_dict.items() 
                          if param.dim() < 2
        ]

        optim_groups = [
            {
                'params': decay_params, 
                'weight_decay': weight_decay
            },
            {
                'params': nodecay_params, 
                'weight_decay': 0.0
            }
        ]
        #-----------------
        num_decay_params = sum( p.numel() for p in decay_params )
        num_nodecay_params = sum( p.numel() for p in nodecay_params )


        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        #-----------------

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters

        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        
        return optimizer
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def compare_dictionaries(dict1, dict2):
    """
    Compares two dictionaries, handling PyTorch tensors.

    Args:
        dict1: The first dictionary.
        dict2: The second dictionary.

    Returns:
        A dictionary containing differences.
    """

    differences = {
        "different_values": {},
        "missing_keys_in_dict1": [],
        "missing_keys_in_dict2": [],
        "extra_keys_in_dict1": [],
        "extra_keys_in_dict2": [],
    }

    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    common_keys = keys1.intersection(keys2)

    for key in common_keys:
        val1 = dict1[key]
        val2 = dict2[key]

        if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            if not torch.equal(val1, val2):  # Use torch.equal for tensor comparison
                differences["different_values"][key] = {
                    "dict1": val1,
                    "dict2": val2,
                }
        elif val1 != val2: # handles non tensor comparisons.
            differences["different_values"][key] = {
                "dict1": val1,
                "dict2": val2,
            }

    differences["missing_keys_in_dict1"] = list(keys2 - keys1)
    differences["missing_keys_in_dict2"] = list(keys1 - keys2)
    differences["extra_keys_in_dict1"] = list(keys1 - keys2)
    differences["extra_keys_in_dict2"] = list(keys2 - keys1)

    return differences
#-------------------------------------------------------------------------

import tiktoken
#-------------------------------------------------------------------------
class DataLoaderLite:
    #-------------------------------------------------------------------------
    def __init__(self, B, T):
        # B - batch size, T - sequence length
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        # **** DELETE THIS CHANGE ***
        # file_path = './models/10_nano_gpt/current/'
        # with open(f'{file_path}input.txt', 'r') as f:
        with open('input.txt', 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def next_batch(self):
        B, T = self.B, self.T
        '''
        We are going to read and use sequentially everything in the file. we will use the curr pos
        to keep track where we left. In the line below we are reading from tokens, starting from where
        we left off last, and then adding the range of B(batch size) + T(sequence len) + 1 as it's non-inclusive
        '''
        buf = self.tokens[self.current_position : self.current_position+B*T+1]

        # 1 token offset. Then reshape it to [B, T] as 'buf' is a 1 dim tensor
        x = (buf[:-1]).view(B, T) # inputs - from idx 0 to the end minus the last token
        y = (buf[1:]).view(B, T) # targets - from the idx 1 to the last token

        # advance the position in the tensor
        self.current_position += B * T

        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------

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
device = get_device()
import time

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

#------------------------------
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 16 # micro batch size
T = 1024 # sequence length
# B = 4
# T = 32 # sequence length

assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"


grad_accum_steps = total_batch_size // (B * T)


print('\n\n-------------------------')

print(f'B: {B}\t T: {T}')
print(f'total_batch_size: {total_batch_size}\t(B * T): {(B * T)}\ntotal_batch_size // (B * T): {grad_accum_steps}\n')


print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
print('-------------------------\n\n')


train_loader = DataLoaderLite(B=B, T=T)
#------------------------------



'''
The 'high' precision setting allows float32 matrix multiplications to use TensorFloat32, which 
has 10 mantissa bits explicitly stored, or to treat each float32 number as the sum of two 
bfloat16 numbers, which provides approximately 16 mantissa bits with 14 bits explicitly stored. 
If the appropriate fast algorithms are not available, the computations fall back to using the 
'highest' precision setting, which employs the full float32 datatype with 24 mantissa bits.

This configuration is particularly beneficial for CUDA devices, where TensorFloat32 operations 
can be significantly faster than traditional float32 operations. The 'high' precision setting 
leverages these faster operations to improve performance without a substantial loss in precision, 
making it a useful option for many machine learning and scientific computing applications.
'''
torch.set_float32_matmul_precision('high')

# get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

# compile - this greatly increase the performance
if torch.cuda.is_available():
    model = torch.compile(model)

max_lr = 6e-4
min_lr = max_lr * 0.1 # 0.000059999999999999995
warmup_steps = 10
max_steps = 50   

max_steps_minus_warmup_steps = max_steps - warmup_steps
max_lr_minus_min_lr = max_lr - min_lr

'''
With a LR scheduler we acknowledge that are different stages during the training process. Initially
  the model is in an almost useless state where it lears more about what tokens are not used. Then after
  it warms up we can make faster (macro) corrections with a larger LR, but once it has achieved a 
  certain level of stability we want to gradually fine tune its learning rate to something smaller
  so corrections are smoother (micro)
'''
#-------------------------------------------------------------------------
def get_lr(it:int): # it -> steps from the training process

    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps: # usually 10
        # 6e-4 * (0 + 1)  / 10  = 6e-4 * 1  / 10 = 6e-4    / 10 = 5.9999999999999995e-05
        # 6e-4 * (1 + 1)  / 10  = 6e-4 * 2  / 10 = 0.0012  / 10 = 0.00011999999999999999
        # ...
        # 6e-4 * (9 + 1)  / 10  = 6e-4 * 10 / 10 = 0.00599 / 10 = 0.0006
        return max_lr * (it+1) / warmup_steps
    
    # 2) if it > lr_decay_iters, return min learning rate
    if it >= max_steps:
        return min_lr # 0.000059999999999999995
    

    # 3) in between, use cosine decay down to min learning rate
    #   (10 - 10) / (50 - 10) = 0 / 40 = 0 
    #   (11 - 10) / (50 - 10) = 1 / 40 = 0.025
    #   ...  -> 0.05 , 0.075 , 0.1, 0.125 , 0.15, ... 0.95 , 0.975
    decay_ratio = (it - warmup_steps) / (max_steps_minus_warmup_steps)

    assert 0 <= decay_ratio <= 1

    # 0.5 * ( 1.0 + math.cos( math.pi * 0.0 ) ) = 0.5 * ( 1.0 + math.cos( 0 ) ) = 
    #     0.5 * ( 1.0 + 1.0 ) = 0.5 * 2 = 1
    # 0.5 * ( 1.0 + math.cos( math.pi * 0.025 ) ) = 0.5 * ( 1.0 + math.cos( 0.07853981633974483 ) ) = 
    #     0.5 * ( 1.0 + 0.996917333733128 ) = 0.5 *  1.996917333733128 = 0.998458666866564
    # 
    # so from the start the values are:
    #   1.0 , 0.998458666866564 , 0.9938441702975689, 0.9861849601988383, ... , 0.001541333133436018
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0

    # min_lr + coeff * (max_lr - min_lr) = 5.9999999999999995e-05 + coeff * ( 0.0005399999999999999 )
    # 5.9999999999999995e-05 + 1.0 * ( 0.0005399999999999999 )               = 0.0005999999999999998
    # 5.9999999999999995e-05 + 0.998458666866564 * ( 0.0005399999999999999 ) = 0.0005991676801079444
    # 
    # from the start the values are:
    #   0.0005999999999999998 , 0.0005991676801079444 , 0.0005966758519606872 , 0.0005925398785073725 , 
    #   ... , 0.00006083231989205545
    return min_lr + coeff * (max_lr_minus_min_lr)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# optimize!

# using hyperparameters from GPT3 - literally following what they published in their paper.
#  this routine uses similar values for the parameters, but not quite the same
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

print(f'train_loader.B * train_loader.T: {train_loader.B * train_loader.T}')

#-------------------------------------------------------------------------
for step in range(max_steps):
    t0 = time.time()

    optimizer.zero_grad()
    loss_accum = 0.0

    #-------------------------------------
    for micro_step in range(grad_accum_steps): # note grad_accum_steps = total_batch_size // (B * T)
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)


        '''
        Cast all operatoins within this block (and device, typically a CUDA enabled device) to bfloat16.
        bfloat16 is particularly desiable because it offers a good balance between precision and large
        improvements in performance (for CUDA devices).
        Note: For your specific CUDA card, check the estimated bfloat16 operations and possible other floating
        points types, as NVIDIA often updates their types and performance estimates.
        '''
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            # forward
            logits, loss = model(x, y)

        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps

        '''
        detach() is used to create a tensor that shares the same data as the original 
        tensor but is detached from the computation graph. 
        prevents the loss_accum tensor from contributing to the gradient computation
        Gradient Accumulation: In scenarios where the total batch size is too large to 
        fit into memory (e.g., due to GPU memory constraints), the training process is 
        split into smaller "micro-batches." The losses from these micro-batches are 
        accumulated over several iterations before performing a single optimization 
        step. The variable loss_accum serves as a running total to accumulate the loss 
        values from each micro-batch.
        '''
        loss_accum += loss.detach()
        loss.backward()
    #-------------------------------------

    '''
    using hyperparameters from GPT3 - literally following what they published in their paper.
     this routine uses similar values for the parameters, but not quite the same    
    Calculate the global norm of the parameters - every gradient from the parameters are squared,
      sum all up, then take the square root 
    In the example below we define that the max_norm is no bigger than 1.0
    One of the reasons to use this, is in the case of bad data batches we would get a high loss, 
      which leads to a high gradient. This is a case of 'model shock'.
    Ideally we want the learning process to be more on the smooth side and not too peaky which can
     lead to instabilities and low convergence
    '''
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:

        if 'lr' not in param_group:
            param_summary = f'*** NOTE - LR IS NOT PRESENT in param_group'
        else:
            param_summary = f"param_group['lr']: {param_group['lr']:.4f} - new LR: {lr:.4f}"

        param_group['lr'] = lr
    
    optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize() # wait for the GPU to finish work

    t1 = time.time()
    dt = t1 - t0 # time difference in seconds

    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt

    print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | {param_summary}")
#-------------------------------------------------------------------------

import sys; sys.exit(0)

# prefix tokens
model.eval()
num_return_sequences = 5
max_length = 30
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#-------------------------------------------------------------------------
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size) - [5, 8, 50257]
        
        # take the logits at the last position. Consider that the original logit shape is [5, 8, 50257]
        #  if you take only the last position from the second dim, as in logits[:, -1, :], what you are
        #  effectively doing is [5, 50257] 
        logits_last = logits[:, -1, :] # (B, vocab_size) - [5, 50257] 

        # get the probabilities
        probs = F.softmax(input = logits_last, dim=-1)

        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(input = probs, k = 50, dim=-1)

        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(input = topk_probs, num_samples = 1) # (B, 1)

        # gather the corresponding indices
        xcol = torch.gather(input = topk_indices, dim = -1, index = ix) # (B, 1)

        # append to the sequence
        x = torch.cat((x, xcol), dim=1)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# print the generated text
for i in range(num_return_sequences): # num_return_sequences = 5
    tokens = x[i, :max_length].tolist() # max_length - 30
    decoded = enc.decode(tokens)
    print(">", decoded)
#-------------------------------------------------------------------------

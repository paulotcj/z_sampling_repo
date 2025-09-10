##########################################################################
##
##  IMPORTS
##
##########################################################################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

# supports debugging, logging, visualization, etc.
global_count1 = 0
##########################################################################
##
##  CLASSES
##
##########################################################################
#-------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    r"""
    the MultiHeadAttention class encapsulates the multi-head attention mechanism commonly used in transformer models. 
    It takes care of splitting the input into multiple attention heads, applying attention to each head, and then combining 
    the results. By doing so, the model can capture various relationships in the input data at different scales, improving 
    the expressive ability of the model 
    """

    #-------------------------------------------------------------------------
    def __init__(self, dim_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        print(f'MultiHeadAttention - __init__')
        
        assert dim_model % num_heads == 0, "d_model must be divisible by num_heads"


        # Initialize dimensions
        self.dim_model  = dim_model              # Model's dimension
        self.num_heads  = num_heads              # Number of attention heads to split the input into
        self.dim_key    = dim_model // num_heads # Dimension of each head's key, query, and value

        # Typically:
        # self.dim_model : 512
        # self.num_heads : 8
        # self.dim_key   : 64

        
        # Linear layers for transforming inputs
        self.weight_query   = nn.Linear(in_features = dim_model, out_features = dim_model) # Query transformation
        self.weight_key     = nn.Linear(in_features = dim_model, out_features = dim_model) # Key transformation
        self.weight_value   = nn.Linear(in_features = dim_model, out_features = dim_model) # Value transformation
        self.weight_output  = nn.Linear(in_features = dim_model, out_features = dim_model) # Output transformation
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def scaled_dot_product_attention(self, query, key, value, mask=None):
        # Calculate attention scores - the attention scores are calculated by taking the dot product of queries and keys
        #   and then scaling by the square root of the key dimension

        # query [64, 8, 100, 64] or [64, 8, 99, 64] 
        # key   [64, 8, 100, 64] or [64, 8, 99, 64] 
        # value [64, 8, 100, 64] or [64, 8, 99, 64] 
        # mask  [64, 1, 1, 100]  or [64, 1, 99, 99]

        # print(f'query.shape: {query.shape}')
        # print(f'key.shape:   {key.shape}')
        # print(f'value.shape: {value.shape}')
        # print(f'mask.shape:  {mask.shape}')
        # print('-------')


        # swap the last two dimensions (-1 is the last dimension, -2 is the second to last)
        key_transpose = key.transpose(-2, -1) # [64, 8, 64, 100] or [64, 8, 64, 99]
        
        sqrt_dim_key = math.sqrt(self.dim_key) # sqrt_dim_key = 8 and self.dim_key = 64
        
        # query is [64, 8, 100, 64] or [64, 8, 99, 64] and key_transpose is [64, 8, 64, 100] or [64, 8, 64, 99]
        #   so the final tenson is [64, 8, 100, 100] or [64, 8, 99, 99]
        #
        # For easy visualization: [64, 8, 100, 64] x [64, 8, 64, 100] = [64, 8, 100, 100]
        #                         [64, 8, 99,  64] x [64, 8, 64, 99 ] = [64, 8, 99,  99 ]
        matmul_query_keytranspose = torch.matmul(input = query, other = key_transpose) # [64, 8, 100, 100] or [64, 8, 99, 99]

        attn_scores = matmul_query_keytranspose / sqrt_dim_key # [64, 8, 100, 100] or [64, 8, 99, 99]

        # print(f'self.dim_key:                    {self.dim_key}')
        # print(f'key_transpose.shape:             {key_transpose.shape}')
        # print(f'sqrt_dim_key:                    {sqrt_dim_key}')
        # print(f'matmul_query_keytranspose.shape: {matmul_query_keytranspose.shape}')
        # print(f'attn_scores.shape:               {attn_scores.shape}')
        
        #--------------------------------
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        #   this is more complicated and hard to debug than it seems
        if mask is not None:

            #------ debug init ------
            # num_true = torch.sum(mask)
            # num_false = mask.numel() - num_true
            # if num_false > 1 and num_true > 1:
            #     print(f'False count in mask: {num_false}')
            #     print(f'True count in mask: {num_true}')
            #     print('====================================')
            #     # mask      [64, 1, 1, 100]  or [64, 1, 99, 99]
            #     print(f'mask[0][0]:\n{mask[0][0]}')
            #     print(f'attn_scores[0][0]:\n{attn_scores[0][0]}')  
            #------ debug end ------
            
            attn_scores = attn_scores.masked_fill( mask = mask == 0, value = -1e9)

            #------ debug init ------
            # if num_false > 1 and num_true > 1:
            #     print(f'attn_scores[0][0] after applying mask:\n{attn_scores[0][0]}')
            #------ debug end ------
        #--------------------------------
        
        # [64, 8, 100, 100] or [64, 8, 99, 99]
        attn_probs = torch.softmax(input = attn_scores, dim = -1) # Softmax is applied to obtain attention probabilities
        
        # Multiply by values to obtain the final output. attn_probs [64, 8, 100, 100] or [64, 8, 99, 99] and
        #   value [64, 8, 100, 64] or [64, 8, 99, 64] , so the result is [64, 8, 100, 64] or [64, 8, 99, 64]
        # For easy visualization: [64, 8, 100, 100] x [64, 8, 100, 64] = [64, 8, 100, 64]
        # Remember that with the first 2 dimensions being the same, the matrix multiplication is
        #  [100, 100] x [100, 64], and the rule for matrix multiplication is A[M,N] x B[N,P] = C[M,P]
        #  so effectively [100, 64]
        output = torch.matmul(input = attn_probs, other = value) # [64, 8, 100, 64] or [64, 8, 100, 64]

        return output # [64, 8, 100, 64] or [64, 8, 100, 64]
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------  
    def split_heads(self, x):
        # Reshape the input x into the shape (batch_size, num_heads, seq_length, dim_key). It enables the model to 
        #   process multiple attention heads concurrently, allowing for parallel computation

        # x shape: [64, 100, 512] or [64, 99, 512]

        #64         (100 or 99)  512
        batch_size, seq_length,  dim_model = x.size()


        # note: dim_key is dim_model // num_heads, considering the dim_model is 512 and 
        #   num_heads is 8, dim_key will be 64, so in this case we are splitting the last
        #   dimension which is 512 into 8 parts of 64 each (without changing the original object x)
        #               64          (100 or 99)      8               64
        x_view = x.view(batch_size, seq_length, self.num_heads, self.dim_key) # [64, 100, 8, 64] or [64, 99, 8, 64]


        # note that x_view is [64, 100, 8, 64] and we perform a transpose operation swapping the  dimensions
        #   1 and 2, so the expected result is that dims 100 and 8 should switch places, and we observe that
        #   the result is in fact [64, 8, 100, 64]
        return_object = x_view.transpose(1, 2) # [64, 8, 100, 64] or [64, 8, 99, 64]


        return return_object # [64, 8, 100, 64] or [64, 8, 99, 64]
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------    
    def combine_heads(self, x):
        # After applying attention to each head separately, this method combines the results back into a single tensor 
        #   of shape (batch_size, seq_length, d_model). This prepares the result for further processing

        # x shape: [64, 8, 100, 64] or [64, 8, 99, 64]


        # batch_size (64), num_heads (8), seq_length (100 or 99), dim_key (64)
        batch_size, _, seq_length, dim_key = x.size() 

        # x_transposed reverts the action of swapping dims from split_heads method, here we are swapping dims
        #  1 and 2, from the input x with shape [64, 8, 100, 64], we expect to have [64, 100, 8, 64]
        x_transposed = x.transpose(1, 2).contiguous() # [64, 100, 8, 64] or [64, 99, 8, 64]


        # now create a view, where we keep the first 2 dimensions but join the last 2 ( 8 * 64 = 512)
        #                                 64          (100 or 99)      512
        return_object = x_transposed.view(batch_size, seq_length, self.dim_model)

        return return_object # [64, 100, 512] or [64, 99, 512]
    #-------------------------------------------------------------------------
    #------------------------------------------------------------------------- 
    def forward(self, query, key, value, mask=None):
        # 1 - Apply Linear Transformations: The queries, keys, and values are first passed through linear transformations 
        #   using the weights defined in the initialization
        # 2 - Split Heads: The transformed query, key, value are split into multiple heads using the split_heads method

        # query [64, 100, 512]
        # key   [64, 100, 512]
        # value [64, 100, 512]
        # mask  [64, 1, 1, 100]



        # linear layers with input:512 output:512 all tensors used here are [64, 100, 512] or [64, 99, 512]
        weight_query = self.weight_query(query) # [64, 100, 512] or [64, 99, 512]
        weight_key   = self.weight_key(key)     # [64, 100, 512] or [64, 99, 512]
        weight_value = self.weight_value(value) # [64, 100, 512] or [64, 99, 512]

        #          all tensors are ->   [64, 100, 512] or [64, 99, 512]
        query   = self.split_heads( x = weight_query )   # [64, 8, 100, 64] or [64, 8, 99, 64]
        key     = self.split_heads( x = weight_key     ) # [64, 8, 100, 64] or [64, 8, 99, 64]
        value   = self.split_heads( x = weight_value )   # [64, 8, 100, 64] or [64, 8, 99, 64]


        
        # Apply Scaled Dot-Product Attention. The scaled_dot_product_attention method is called on the split heads
        #   query key and valye are -> [64, 8, 100, 64] or [64, 8, 99, 64]            mask -> [64, 1, 1, 100]
        attn_output = self.scaled_dot_product_attention(query = query, key = key, value = value, mask = mask) # [64, 8, 100, 64] or [64, 8, 99, 64]


        # Combine Heads - The results from each head are combined back into a single tensor using the combine_heads method
        combine_heads_attn_output = self.combine_heads(x = attn_output) # [64, 100, 512] or [64, 100, 512]

        # apply weights from a Linear Layer - this layer has input:512 output:512
        output = self.weight_output(combine_heads_attn_output) # [64, 100, 512])

        return output
    #-------------------------------------------------------------------------
#------------------------------------------------------------------------- 
#-------------------------------------------------------------------------
class PositionWiseFeedForward(nn.Module):
    """
    the PositionWiseFeedForward class defines a position-wise feed-forward neural network that consists of two linear 
    layers with a ReLU activation function in between. In the context of transformer models, this feed-forward network 
    is applied to each position separately and identically. It helps in transforming the features learned by the 
    attention mechanisms within the transformer, acting as an additional processing step for the attention outputs
    """
    #-------------------------------------------------------------------------
    def __init__(self, dim_model, dim_feedforward):
        super(PositionWiseFeedForward, self).__init__()
        r"""
        dim_model: Dimensionality of the model's input and output
        dim_feedforward: Dimensionality of the inner layer in the feed-forward network
        """
        print('PositionWiseFeedForward - __init__')

        # dim_model:       512
        # dim_feedforward: 2048

       
        # self.full_conn_layer_1 and self.full_conn_layer_2 - two fully connected (linear) layers 
        #   with input and output dimensions as defined by dim_model and dim_feedforward
        self.full_conn_layer_1  = nn.Linear( in_features = dim_model, out_features = dim_feedforward ) # 512 , 2048
        self.full_conn_layer_2  = nn.Linear( in_features = dim_feedforward, out_features = dim_model ) # 2048, 512
        self.relu = nn.ReLU() # activation function, which introduces non-linearity between the two linear layers
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, x):
        # x [64, 100, 512] or [64, 99, 512] - input to the feed-forward network (batch_size, seq_length, dim_model)

        #-----------------------
        """
        self.full_conn_layer_1(x): The input is first passed through the first linear layer (fc1).
        self.relu(...): The output of fc1 is then passed through a ReLU activation function. ReLU replaces 
          all negative values with zeros, introducing non-linearity into the model.
        self.full_conn_layer_2(...): The activated output is then passed through the second linear layer (fc2), producing 
          the final output.    
        """
        #-----------------------


        """
        self.full_conn_layer_1 uses in_features as dim_model (512) and out_features as dim_feedforward (2048)
         all parameters provided in the constructor
        Note that x has shape [64, 100, 512], so the linear layer will be applied to the last dimension, in other
          words, we will send the last dimension with 512 elements, 100 times as per the second dimension, and
          64 times as per the first dimension, and the output per iteration of 512 units is 512, so the final
          output will be [64, 100, 2048]
                                              [64, 100, 512]"""
        result_fcl_1 = self.full_conn_layer_1(input = x) # [64, 100, 2048] or [64, 99, 2048]

        result_relu_fcl_1 = self.relu(input = result_fcl_1) # [64, 100, 2048] or [64, 99, 2048] - standard ReLU
        
        # We have the same a smiliar situation as above, but now we are going from 2048 to 512
        # self.full_conn_layer_2 uses in_features as dim_feedforward (2048) and out_features as dim_model (512)
        #                                             [64, 100, 2048] or [64, 99, 2048]
        result_fcl_2 = self.full_conn_layer_2(input = result_relu_fcl_1) # [64, 100, 512] or [64, 99, 512]
        #-----------------------

            
        return result_fcl_2 # [64, 100, 512] or [64, 99, 512]
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------   
#-------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    PositionalEncoding class adds information about the position of tokens within the sequence. Since the 
    transformer model lacks inherent knowledge of the order of tokens (due to its self-attention mechanism), 
    this class helps the model to consider the position of tokens in the sequence. The sinusoidal functions 
    used are chosen to allow the model to easily learn to attend to relative positions, as they produce a 
    unique and smooth encoding for each position in the sequence. The class is implemented as a nn.Module, 
    allowing it to be used as a layer within the model. The forward method adds the positional encodings to 
    the input tensor x, which is expected to have shape (batch_size, seq_length, d_model).    
    """
    #-------------------------------------------------------------------------
    def __init__(self, dim_model_input, max_seq_length):
        super(PositionalEncoding, self).__init__()
        # dim_model_input: The dimension of the model's input - 512
        # max_seq_length: The maximum length of the sequence for which positional encodings are pre-computed - 100

        print(f'\nPositionalEncoding - __init__')


        #                           100             512                torch.Size([100, 512])
        pos_encodings = torch.zeros(max_seq_length, dim_model_input) # tensor filled with zeros, which will be populated with positional encodings

        #---------
        #                                        100                                               torch.Size([100, 1])
        position = torch.arange(start = 0, end = max_seq_length, dtype=torch.float).unsqueeze(1) # tensor containing the position indices for each position in the sequence
        # generates a 1-dimensional tensor, from 0 to max_seq_length -1 (99). Then unsqueeze(1) adds 
        #   a dimension at index 1. So if originally the tensor was [0.0, 1.0, 2.0, ..., 99.0], now is: [[0.0], [1.0], [2.0], ..., [99.0]]
        #---------

        #---------
        temp_value1 = (math.log(10000.0) / dim_model_input) * -1 # value used to scale the position indices in a specific way (-0.017988946039015984)

        # new tensor with len 256 with elements resulting from the exponentiation of 'e'. e.g.: torch.exp(torch.tensor([1, 2, 3])) -> 
        #   (e^1, e^2, e^3) -> tensor([ 2.7183,  7.3891, 20.0855])
        div_term = torch.exp(
            # tensor                      512  (efffectively 256 len)   
            torch.arange(start = 0, end = dim_model_input, step = 2).float() *  temp_value1
        ) # term used to scale the position indices in a specific way
        #---------
        

        #             [100,1]    [256]  (both torch.Tensor) -> [100, 256]
        temp_value2 = position * div_term


        
        
        #----------------------------
        # the sine function is applied to the even indices and the cosine function to the odd indices of pe
        # note that temp_value has shape of [100,256] and pos_encodings has shape of [100,512], so we will
        #   be using the same number twice, but in one case we will use sin and in the other cos
        # ---------
        #   all the rows as denoted by [ :, ... ] then from column 0 to its lenght, use 2 as step, denoted by
        #   [ ... , 0::2]
        pos_encodings[:, 0::2] = torch.sin(temp_value2)

        #  all the rows as denoted by [ :, ... ] then from column 1 to its lenght, use 2 as step, denoted by
        #   [ ... , 1::2]
        pos_encodings[:, 1::2] = torch.cos(temp_value2)
        #----------------------------

        #-------------------------------------------------------------------------
        def data_check():
            print('----')
            print(f'temp_value2[0][0]: {temp_value2[0][0]}')   # 0.0
            print(f'temp_value2[0][1]: {temp_value2[0][1]}')   # 0.0
            print(f'temp_value2[0][2]: {temp_value2[0][2]}')   # 0.0
            print(f'temp_value2[0][3]: {temp_value2[0][3]}')   # 0.0
            print(f'temp_value2[0][98]: {temp_value2[0][98]}') # 0.0
            print(f'temp_value2[0][99]: {temp_value2[0][99]}') # 0.0
            print('----')
            print(f'temp_value2[1][0]: {temp_value2[1][0]}')   # 1.0
            print(f'temp_value2[1][1]: {temp_value2[1][1]}')   # 0.9646615982055664
            print(f'temp_value2[1][2]: {temp_value2[1][2]}')   # 0.9305720329284668
            print(f'temp_value2[1][3]: {temp_value2[1][3]}')   # 0.8976871371269226
            print(f'temp_value2[1][98]: {temp_value2[1][98]}') # 0.02942727319896221
            print(f'temp_value2[1][99]: {temp_value2[1][99]}') # 0.02838735654950142
            print('----')
            print(f'temp_value2[2][0]: {temp_value2[2][0]}')   # 2.0
            print(f'temp_value2[2][1]: {temp_value2[2][1]}')   # 1.9293231964111328
            print(f'temp_value2[2][2]: {temp_value2[2][2]}')   # 1.8611440658569336
            print(f'temp_value2[2][3]: {temp_value2[2][3]}')   # 1.7953742742538452
            print(f'temp_value2[2][98]: {temp_value2[2][98]}') # 0.05885454639792442
            print(f'temp_value2[2][99]: {temp_value2[2][99]}') # 0.05677471309900284            
            print('----')
            print(f'pos_encodings[0][0]: {pos_encodings[0][0]}') # 0.0
            print(f'pos_encodings[0][1]: {pos_encodings[0][1]}') # 1.0
            print(f'pos_encodings[0][2]: {pos_encodings[0][2]}') # 0.0
            print(f'pos_encodings[0][3]: {pos_encodings[0][3]}') # 1.0
            print('----')
            print(f'pos_encodings[1][0]: {pos_encodings[1][0]}') # 0.8414709568023682
            print(f'pos_encodings[1][1]: {pos_encodings[1][1]}') # 0.5403023362159729
            print(f'pos_encodings[1][2]: {pos_encodings[1][2]}') # 0.8218562006950378
            print(f'pos_encodings[1][3]: {pos_encodings[1][3]}') # 0.5696950554847717
            print('----')
            print(f'pos_encodings[2][0]: {pos_encodings[2][0]}')   # 0.9092974066734314
            print(f'pos_encodings[2][1]: {pos_encodings[2][1]}')   # -0.416146844625473
            print(f'pos_encodings[2][2]: {pos_encodings[2][2]}')   # 0.9364147782325745
            print(f'pos_encodings[2][3]: {pos_encodings[2][3]}')   # -0.3508951663970947
            print(f'pos_encodings[2][98]: {pos_encodings[2][98]}') # 0.33639633655548096
            print(f'pos_encodings[2][99]: {pos_encodings[2][99]}') # 0.9417204856872559
        #-------------------------------------------------------------------------
        # data_check()
    


        # [1, 100, 512]
        pos_encodings = pos_encodings.unsqueeze(0) # from originally [100, 512] to [1, 100, 512]

        # pe is registered as a buffer, which means it will be part of the module's state but will not be considered a trainable parameter
        self.register_buffer(name = 'pe', tensor = pos_encodings)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, x):
        # The forward method simply adds the positional encodings to the input x. It uses the first x.size(1) 
        #   elements of pe to ensure that the positional encodings match the actual sequence length of x

        # x shape: [64, 100, 512] or [64, 99, 512]
        # pe shape: [1, 100, 512] - note pe is pos_encodings


        # this is a 3D tensor, so we select all elements from the first dim [:, ...], then we select
        #  the first 100 elements from the second dim [..., :x,size(1)], which is typically 100, but
        #  since this dimension has a len of 100, we end up selecting all elements, and then we don't
        #  specify the third, thus selecting it all
        #                     100
        temp_pe = self.pe[:, :x.size(1)] # [1, 100, 512] or [1, 99, 512]


        # given two objects shape are mismatched, pytorch will perform a broadcasting operation
        return_obj = x + temp_pe # [64, 100, 512] or [64, 99, 512]

        return return_obj
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------   
#-------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """
    The EncoderLayer class defines a single layer of the transformer's encoder. It encapsulates a multi-head 
    self-attention mechanism followed by position-wise feed-forward neural network, with residual connections, 
    layer normalization, and dropout applied as appropriate. These components together allow the encoder to 
    capture complex relationships in the input data and transform them into a useful representation for 
    downstream tasks. Typically, multiple such encoder layers are stacked to form the complete encoder part 
    of a transformer model    
    """
    #-------------------------------------------------------------------------
    def __init__(self, dim_model, num_heads, dim_feedforward, dropout):
        super(EncoderLayer, self).__init__()
        r"""
        dim_model: The dimensionality of the input. 
        num_heads: The number of attention heads in the multi-head attention.
        dim_feedforward: The dimensionality of the inner layer in the position-wise feed-forward network.
        dropout: The dropout rate used for regularization
        """
        # dim_model:       512
        # num_heads:       8
        # dim_feedforward: 2048
        # dropout:         0.1

        print('EncoderLayer - __init__')


        self.self_mult_head_attn = MultiHeadAttention(dim_model = dim_model, num_heads = num_heads) # Multi-head attention mechanism
        self.feed_forward = PositionWiseFeedForward(dim_model = dim_model, dim_feedforward = dim_feedforward) # Position-wise feed-forward neural network
        self.layer_norm1  = nn.LayerNorm(normalized_shape = dim_model) # self.layer_norm1 and self.layer_norm2: Layer normalization, applied to smooth the layer's input
        self.layer_norm2  = nn.LayerNorm(normalized_shape = dim_model) # see above
        self.dropout      = nn.Dropout(p = dropout)   # Dropout layer, used to prevent overfitting by randomly setting some activations to zero during training
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------        
    def forward(self, x, mask):
        # x [64, 100, 512]: The input to the encoder layer. 
        # mask [64, 1, 1, 100]: Optional mask to ignore certain parts of the input. 

        print('>>> EncoderLayer - forward')


        
        # Self-Attention: The input x is passed through the multi-head self-attention mechanism
        attn_output = self.self_mult_head_attn( # [64, 100, 512]
            query = x,    # [64, 100, 512]
            key   = x,    # [64, 100, 512]
            value = x,    # [64, 100, 512]
            mask  = mask  # [64, 1, 1, 100]
        )

        

        # Add & Normalize (after Attention): The attention output is added to the original input
        #   (residual connection), followed by dropout and normalization using norm1
        attn_output_dropout = self.dropout(input = attn_output) # [64, 100, 512]
        x_temp = x + attn_output_dropout # [64, 100, 512] -> ( x[64, 100, 512] + attn_output_dropout[64, 100, 512] )
        x = self.layer_norm1( input = x_temp ) # [64, 100, 512]


        
        
        # Feed-Forward Network: The output from the previous step is passed through the 
        #   position-wise feed-forward network
        ff_output = self.feed_forward( x = x ) # [64, 100, 512])
        

        

        # Add & Normalize (after Feed-Forward): Similar to step 2, the feed-forward output is 
        #   added to the input of this stage (residual connection), followed by dropout and 
        #   normalization using norm2        
        ff_output_dropout = self.dropout(input = ff_output) # [64, 100, 512]
        x_temp = x + ff_output_dropout # [64, 100, 512]
        x = self.layer_norm2( input = x_temp ) # [64, 100, 512]



        # [64, 100, 512]
        return x # Output: The processed tensor is returned as the output of the encoder layer
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------   
#-------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    """
    The DecoderLayer class defines a single layer of the transformer's decoder. It consists of a multi-head 
    self-attention mechanism, a multi-head cross-attention mechanism (that attends to the encoder's output), 
    a position-wise feed-forward neural network, and the corresponding residual connections, layer 
    normalization, and dropout layers. This combination enables the decoder to generate meaningful outputs 
    based on the encoder's representations, taking into account both the target sequence and the source 
    sequence. As with the encoder, multiple decoder layers are typically stacked to form the complete 
    decoder part of a transformer model.    
    """
    #-------------------------------------------------------------------------
    def __init__(self, dim_model, num_heads, dim_feedforward, dropout):
        super(DecoderLayer, self).__init__()

        # dim_model: The dimensionality of the input.
        # num_heads: The number of attention heads in the multi-head attention.
        # dim_feedforward: The dimensionality of the inner layer in the feed-forward network.
        # dropout: The dropout rate for regularization.

        print('DecoderLayer - __init__')

        self.self_mult_head_attn  = MultiHeadAttention(dim_model = dim_model, num_heads = num_heads) # Multi-head self-attention mechanism for the target sequence
        self.cross_mult_head_attn = MultiHeadAttention(dim_model = dim_model, num_heads = num_heads) # Multi-head attention mechanism that attends to the encoder's output
        self.feed_forward = PositionWiseFeedForward(dim_model, dim_feedforward) # Position-wise feed-forward neural network
        self.norm_layer1  = nn.LayerNorm(dim_model) # Layer normalization components
        self.norm_layer2  = nn.LayerNorm(dim_model) # same as above
        self.norm_layer3  = nn.LayerNorm(dim_model) # same as above
        self.dropout      = nn.Dropout(dropout)     # Dropout layer for regularization
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        x: The input to the decoder layer.
        enc_output: The output from the corresponding encoder (used in the cross-attention step).
        src_mask: Source mask to ignore certain parts of the encoder's output.
        tgt_mask: Target mask to ignore certain parts of the decoder's input
        """
        
        print('<<< DecoderLayer - forward')

        # x [64, 99, 512]
        # enc_output [64, 100, 512]
        # src_mask [64, 1, 1, 100]
        # tgt_mask [64, 1, 99, 99]

        # Self-Attention on Target Sequence: The input x is processed through a self-attention mechanism.
        # [64, 99, 512]
        attn_output = self.self_mult_head_attn( 
            query = x, # [64, 99, 512]
            key   = x, # [64, 99, 512]
            value = x, # [64, 99, 512]
            mask  = tgt_mask # [64, 1, 99, 99]
        ) 

        # print(f'x.shape:            {x.shape}')
        # print(f'tgt_mask.shape:     {tgt_mask.shape}')

        # print(f'attn_output.shape:  {attn_output.shape}')
        # exit()
        
        #----------------- start
        # Add & Normalize (after Self-Attention): The output from self-attention is added to the 
        #   original x, followed by dropout and normalization using norm1.

        #                                        [64, 99, 512]
        attn_output_dropout         = self.dropout(attn_output)   # [64, 99, 512]
        
        # x [64, 99, 512]      attn_output_dropout [64, 99, 512]
        attn_output_dropout_plus_x  = x + attn_output_dropout     # [64, 99, 512]

        #                             [64, 99, 512]
        x = self.norm_layer1( input = attn_output_dropout_plus_x) # [64, 99, 512]
        #----------------- end


        # Cross-Attention with Encoder Output: The normalized output from the previous step is 
        #   processed through a cross-attention mechanism that attends to the encoder's output enc_output.
        # [64, 99, 512])
        attn_output = self.cross_mult_head_attn( 
            query = x,          # [64, 99, 512]
            key   = enc_output, # [64, 100, 512]
            value = enc_output, # [64, 100, 512]
            mask  = src_mask    # [64, 1, 1, 100]
        )


        #----------------- start
        # Add & Normalize (after Cross-Attention): The output from cross-attention is added to the 
        #   input of this stage, followed by dropout and normalization using norm2.

        #                                  [64, 99, 512]
        attn_output_dropout = self.dropout(attn_output) # [64, 99, 512]

        #             x [64, 99, 512]  attn_output_dropout [64, 99, 512]
        attn_output_dropout_plus_x = x + attn_output_dropout # [64, 99, 512]

        #                             [64, 99, 512]
        x = self.norm_layer2( input = attn_output_dropout_plus_x ) # [64, 99, 512])

        #----------------- end

        # Feed-Forward Network: The output from the previous step is passed through the feed-forward network
        #                              [64, 99, 512]
        ff_output = self.feed_forward( x = x ) # [64, 99, 512]

        
        #----------------- start
        # Add & Normalize (after Feed-Forward): The feed-forward output is added to the input of this stage, 
        #   followed by dropout and normalization using norm3.
                
        #                                [64, 99, 512]
        ff_output_dropout = self.dropout(ff_output) # [64, 99, 512]

        #        x [64, 99, 512]    ff_output_dropout [64, 99, 512]
        ff_output_dropout_plus_x = x + ff_output_dropout # [64, 99, 512]

        #                             [64, 99, 512]
        x = self.norm_layer3( input = ff_output_dropout_plus_x ) # [64, 99, 512]
        #----------------- end

        return x # [64, 99, 512] Output: The processed tensor is returned as the output of the decoder layer.
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------   
#-------------------------------------------------------------------------
class Transformer(nn.Module):
    """
    The Transformer class brings together the various components of a Transformer model, including the 
      embeddings, positional encoding, encoder layers, and decoder layers. It provides a convenient 
      interface for training and inference, encapsulating the complexities of multi-head attention, 
      feed-forward networks, and layer normalization.
    This implementation follows the standard Transformer architecture, making it suitable for 
      sequence-to-sequence tasks like machine translation, text summarization, etc. The inclusion of 
      masking ensures that the model adheres to the causal dependencies within sequences, ignoring 
      padding tokens and preventing information leakage from future tokens.
    These sequential steps empower the Transformer model to efficiently process input sequences and 
      produce corresponding output sequences    
    """
    #-------------------------------------------------------------------------
    def __init__(self, src_vocab_size, tgt_vocab_size, dim_model_embeddings, num_heads, 
                 num_layers, dim_inner_feedforward, max_seq_length, dropout):
        super(Transformer, self).__init__()

        """
        src_vocab_size: Source vocabulary size                                               - 5000
        tgt_vocab_size: Target vocabulary size                                               - 5000
        dim_model_embeddings: The dimensionality of the model's embeddings                   - 512
        num_heads: Number of attention heads in the multi-head attention mechanism           - 8
        num_layers: Number of layers for both the encoder and the decoder                    - 6
        dim_inner_feedforward: Dimensionality of the inner layer in the feed-forward network - 2048
        max_seq_length: Maximum sequence length for positional encoding                      - 100
        dropout: Dropout rate for regularization                                             - 0.1
        """

        print('\n\nTransformer - __init__')
        

        self.encoder_embedding = nn.Embedding( # Embedding layer for the source sequence - Embedding(5000, 512)
            num_embeddings  = src_vocab_size,       #5000
            embedding_dim   = dim_model_embeddings  #512
        )    

        

        self.decoder_embedding = nn.Embedding( # Embedding layer for the target sequence - Embedding(5000, 512)
            num_embeddings  = tgt_vocab_size,       #5000
            embedding_dim   = dim_model_embeddings  #512
        )
          

        
        self.positional_encoding = PositionalEncoding(  # Positional encoding component
            dim_model_input = dim_model_embeddings, # 512
            max_seq_length  = max_seq_length        # 100
        )  

        #-------
        list_encoder_layers = []
        list_decoder_layers = []

        print('\n-------------')
        print('Loop object creation')
        print('-------------')
        for _ in range(num_layers):
            
            temp = EncoderLayer(
                dim_model       = dim_model_embeddings,  # 512
                num_heads       = num_heads,             # 8
                dim_feedforward = dim_inner_feedforward, # 2048
                dropout         = dropout                # 0.1
            )
            list_encoder_layers.append( temp )
            print('---')
            temp = DecoderLayer(
                dim_model       = dim_model_embeddings,  # 512
                num_heads       = num_heads,             # 8
                dim_feedforward = dim_inner_feedforward, # 2048
                dropout         = dropout                # 0.1
            ) 
            list_decoder_layers.append( temp )
            
        print('-------------')

        self.encoder_layers = nn.ModuleList(modules = list_encoder_layers) # A list of encoder layers
        self.decoder_layers = nn.ModuleList(modules = list_decoder_layers) # A list of decoder layers
        #-------

        self.fully_connected_layer = nn.Linear( in_features = dim_model_embeddings, out_features = tgt_vocab_size ) # Final fully connected (linear) layer mapping to target vocabulary size
        self.dropout = nn.Dropout( dropout ) # Dropout layer
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def generate_mask(self, source_data, target_data):
        """
        This method is used to create masks for the source and target sequences, ensuring that padding 
        tokens are ignored and that future tokens are not visible during training for the target sequence
        """

        # source_data : [64,100] -> [batch_size, max_seq_length]
        # target_data : [64,99] -> [batch_size, max_seq_length - 1]

        """
        (source_data != 0) -> creates a tensor where each element is True if the corresponding 
          element is not equal to 0 (and false otherwise)
        .unsqueeze(1) -> adds a dimension at index 1
        .unsqueeze(2) -> adds a dimension at index 2
        """
        src_mask      = (source_data != 0).unsqueeze(1).unsqueeze(2) # [64, 1, 1, 100]
        tgt_mask_temp = (target_data != 0).unsqueeze(1).unsqueeze(3) # [64, 1, 99, 1]
        seq_length    = target_data.size(1)                          # [64,99] idx 1 -> 99


        """
        e.g.: torch.ones(6,6) ->
            tensor([[1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1.]])
        """
        ones_3d_matrix = torch.ones(1, seq_length, seq_length)         # [1, 99, 99]

        """
        e.g.: orch.ones(6,6) -> torch.triu(ones_matrix, diagonal=1)
        tensor([[0., 1., 1., 1., 1., 1.],
                [0., 0., 1., 1., 1., 1.],
                [0., 0., 0., 1., 1., 1.],
                [0., 0., 0., 0., 1., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0.]]) 

        The diagonal 0 is the main diagonal, 1 is the diagonal above the main diagonal  
        """
        upper_triangular_mask = torch.triu(ones_3d_matrix, diagonal=1) # [1, 99, 99]

        """
        this will transform the upper_triangular_mask, the new values are now True or False, and
          where it was 0 before now it's True, and where it was 1 before now it's False
        
        e.g.: (1 - upper_triangular_mask) flips the values
        tensor([[0., 1., 1., 1., 1., 1.],
                [0., 0., 1., 1., 1., 1.],
                [0., 0., 0., 1., 1., 1.],
                [0., 0., 0., 0., 1., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0.]])

        (1 - upper_triangular_mask) -> 
        tensor([[1., 0., 0., 0., 0., 0.],
                [1., 1., 0., 0., 0., 0.],
                [1., 1., 1., 0., 0., 0.],
                [1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 0.],
                [1., 1., 1., 1., 1., 1.]])

        .bool() -> converts the tensor to boolean values
        tensor([[ True,  False, False, False, False, False],
                [ True,  True,  False, False, False, False],
                [ True,  True,  True,  False, False, False],
                [ True,  True,  True,  True,  False, False],
                [ True,  True,  True,  True,  True,  False],
                [ True,  True,  True,  True,  True,  True]])        

        """
        no_peak_mask = (1 - upper_triangular_mask).bool() # [1, 99, 99]



        """
        tgt_mask_temp shape -> [64, 1, 99, 1] , no_peak_mask shape -> [1, 99, 99], 
        tgt_mask shape -> [64, 1, 99, 99]

        In this case pytorch performs a broadcasting operation, consider that 
        no_peak_mask shape -> [1, 99, 99] and tgt_mask_temp shape -> [64, 1, 99, 1]
        pytorch will use no_peak_mask shape -> [1, 99, 99] and tgt_mask_temp shape -> [xx, 1, 99, 1]
        and then the broadcasting will be [xx, 1, 99, 99]
        Sinde xx is 64, the final shape will be [64, 1, 99, 99]
        """
        tgt_mask = tgt_mask_temp & no_peak_mask # [64, 1, 99, 99]


        #-------------------------------------------------------------------------
        def check_data():

            print(f'\n\nsrc_mask shape: {src_mask.shape}')
            print(f'tgt_mask_temp shape: {tgt_mask_temp.shape}')
            print(f'seq_length: {seq_length}')
            
            print('\n\n')
            

            print(f'ones_3d_matrix shape: {ones_3d_matrix.shape}')
            for k, v in enumerate(ones_3d_matrix):
                print(f'k: {k} - \n{v}')

            print(f'\n\nupper_triangular_mask shape: {upper_triangular_mask.shape}')
            for k, v in enumerate(upper_triangular_mask):
                print(f'k: {k} - \n{v}')


            print(f'\n\nnopeak_mask shape: {no_peak_mask.shape}')
            for k, v in enumerate(no_peak_mask):
                print(f'k: {k} - \n{v}')

            print(f'\n\ntgt_mask shape: {tgt_mask.shape}')
            for k, v in enumerate(tgt_mask):
                print(f'k: {k} - \n{v}')

            print('\n\n')
            print(f'tgt_mask_temp shape: {tgt_mask_temp.shape}')
            print(f'nopeak_mask shape: {no_peak_mask.shape}')
            print(f'tgt_mask shape: {tgt_mask.shape}')
        #-------------------------------------------------------------------------
        # check_data()
      
        #      [64, 1, 1, 100]    [64, 1, 99, 99]
        return src_mask,          tgt_mask
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, source_data, target_data):
        """
        Input Embedding and Positional Encoding: The source and target sequences are first embedded using 
          their respective embedding layers and then added to their positional encodings.
        Encoder Layers: The source sequence is passed through the encoder layers, with the final encoder 
          output representing the processed source sequence.
        Decoder Layers: The target sequence and the encoder's output are passed through the decoder layers, 
          resulting in the decoder's output.
        Final Linear Layer: The decoder's output is mapped to the target vocabulary size using a fully 
          connected (linear) layer.        
        """

        print('Transformer - forward')

        # source_data: [64, 100]
        # target_data: [64, 99]

        #------------------
        # temp and aux variables
        src_dt_embedded = self.encoder_embedding(source_data) # [64, 100, 512] -> 512 = dim_model_embeddings
        src_dt_emb_pos  = self.positional_encoding(src_dt_embedded) # [64, 100, 512] -> 512 = dim_model_embeddings
        #----
        tg_dt_embedded = self.decoder_embedding(target_data) # [64, 99, 512]
        tg_dt_emb_pos = self.positional_encoding(tg_dt_embedded) # [64, 99, 512]

        #------------------
        # masks and vars at the end of the pipeline
        # src_mask: [64, 1, 1, 100] , tgt_mask: [64, 1, 99, 99]
        src_mask, tgt_mask  = self.generate_mask(source_data = source_data, target_data = target_data)
        src_embedded        = self.dropout( input = src_dt_emb_pos ) # [64, 100, 512]
        tgt_embedded        = self.dropout( input = tg_dt_emb_pos )  # [64, 99, 512]

        
        #------------------
        """
        Note:
            temp = EncoderLayer(
                dim_model       = dim_model_embeddings,  # 512
                num_heads       = num_heads,             # 8
                dim_feedforward = dim_inner_feedforward, # 2048
                dropout         = dropout                # 0.1
            )
        this class is implemented above
        """
        enc_output = src_embedded # [64, 100, 512] -> [batch_size, max_seq_length, dim_model_embeddings]
        for enc_layer in self.encoder_layers: # len 6
            enc_output = enc_layer(   # [64, 100, 512]
                x       = enc_output, # [64, 100, 512]
                mask    = src_mask    # [64, 1, 1, 100]
            )
        #------------------
        

        #------------------
        """
        Note:
            temp = DecoderLayer(
                dim_model       = dim_model_embeddings,  # 512
                num_heads       = num_heads,             # 8
                dim_feedforward = dim_inner_feedforward, # 2048
                dropout         = dropout                # 0.1
            )
        this class is implemented above
        """
        dec_output = tgt_embedded # [64, 99, 512]
        for dec_layer in self.decoder_layers: # len 6
            dec_output = dec_layer(       # [64, 99, 512]
                x           = dec_output, # [64, 99, 512]
                enc_output  = enc_output, # [64, 100, 512]
                src_mask    = src_mask,   # [64, 1, 1, 100]
                tgt_mask    = tgt_mask    # [64, 1, 99, 99]
            )
        #------------------

        
        output = self.fully_connected_layer(input = dec_output) # [64, 99, 5000] -> [batch_size, max_seq_length-1, tgt_vocab_size]

        return output # output is a tensor representing the model's predictions for the target sequence
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------   

##########################################################################
##
##  PART 1
##
##########################################################################
#-------------------------------------------------------------------------   
def sample_data_preparation():
    """
    Initialize a transformer model and generate random source and target sequences that can be fed 
      into the model. The chosen hyperparameters determine the specific structure and properties 
      of the transformer. This setup could be part of a larger script where the model is trained 
      and evaluated on actual sequence-to-sequence tasks, such as machine translation or text 
      summarization    
    """
    src_vocab_size  = 5000
    tgt_vocab_size  = 5000
    d_model         = 512 # Dimensionality of the model's embeddings, set to 512
    num_heads       = 8
    num_layers      = 6 # Number of layers for both the encoder and the decoder
    d_ff            = 2048 # Dimension of the inner layer in the feed-forward network
    max_seq_length  = 100
    dropout         = 0.1
    batch_size      = 64
    min_value       = 1


    transformer = Transformer(
        src_vocab_size          = src_vocab_size, 
        tgt_vocab_size          = tgt_vocab_size, 
        dim_model_embeddings    = d_model, 
        num_heads               = num_heads, 
        num_layers              = num_layers, 
        dim_inner_feedforward   = d_ff, 
        max_seq_length          = max_seq_length, 
        dropout                 = dropout
    )



    # Generate random sample data
    #  Random integers between 1 and tgt_vocab_size, representing a batch of target sequences with shape (64, max_seq_length)
    #                               1                 5000                    64          100
    src_data = torch.randint( low = min_value, high = src_vocab_size, size = (batch_size, max_seq_length) )  # (batch_size, seq_length) [64,100]
    tgt_data = torch.randint( low = min_value, high = tgt_vocab_size, size = (batch_size, max_seq_length) )  # (batch_size, seq_length) [64,100]

    
    return {
        'transformer'       : transformer,
        'src_data'          : src_data,        # [64,100]
        'tgt_data'          : tgt_data,        # [64,100]
        'tgt_vocab_size'    : tgt_vocab_size,  # 5000
        'src_vocab_size'    : src_vocab_size,  # 5000
        'max_seq_length'    : max_seq_length,  # 100
        'batch_size'        : batch_size       # 64
    }
#-------------------------------------------------------------------------  
#-------------------------------------------------------------------------  
def execute_part1():
    result_sample_data_preparation =  sample_data_preparation()
    return {
        'transformer'   : result_sample_data_preparation['transformer'],
        'src_data'      : result_sample_data_preparation['src_data'],       # [64,100]
        'tgt_data'      : result_sample_data_preparation['tgt_data'],       # [64,100]
        'tgt_vocab_size': result_sample_data_preparation['tgt_vocab_size'], # 5000
        'src_vocab_size': result_sample_data_preparation['src_vocab_size'], # 5000
        'max_seq_length': result_sample_data_preparation['max_seq_length']  # 100
    }

#-------------------------------------------------------------------------  
result_part1 = execute_part1()
##########################################################################
##
##  PART 2
##
##########################################################################
#-------------------------------------------------------------------------  
def training(transformer, src_data, tgt_data, tgt_vocab_size):
    """
    This code snippet trains the transformer model on randomly generated source and target sequences 
    for 100 epochs. It uses the Adam optimizer and the cross-entropy loss function. The loss is 
    printed for each epoch, allowing you to monitor the training progress. In a real-world scenario, 
    you would replace the random source and target sequences with actual data from your task, such 
    as machine translation
    """
    
    # src_data: [64, 100] -> [batch_size, max_seq_length]
    # tgt_data: [64, 100] -> [batch_size, max_seq_length]
    
    
    # Defines the loss function as cross-entropy loss. The ignore_index argument is set to 0, meaning 
    #   the loss will not consider targets with an index of 0 (typically reserved for padding tokens)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    r"""
    Defines the optimizer as Adam with a learning rate of 0.0001 and specific beta values
    b1 Controls the exponential decay rate for the moving average of the gradients  m_t
    b2 Controls the exponential decay rate for the moving average of the squared gradients  v_t
    Betas default values are (0.9, 0.999) - for noisy gradients try to reduce b1 to 0.8, and for
    sparse data decrease b2 to 0.9
    eps is a very small number to prevent division by zero
    """
    optimizer = optim.Adam(params = transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)



    # Sets the transformer model to training mode, enabling behaviors like dropout that only apply during training
    transformer.train()

    training_epochs = 100
    training_epochs = 1
    #------------------------------
    for epoch in range(training_epochs):
        optimizer.zero_grad() # Clears the gradients from the previous iteration

        """
        Passes the source data and the target data (excluding the last token in each sequence) through the 
          transformer. This is common in sequence-to-sequence tasks where the target is shifted by one token
        So with the explanation above in mind note that tgt_data[:, :-1] means, select all rows and all columns
          except the last one
        Consider the example:
        tgt_data
        tensor([[3933, 2734,  325,  ..., 2672, 2453,  742],
                [1992, 2432,  904,  ...,  805, 1234,  690],
                [1562, 3574, 3254,  ..., 3698,  510, 3429],
                ...,
                [1542, 4150, 2908,  ..., 4395,  401, 2284],
                [2390, 3349,  358,  ..., 3117,  710, 2540],
                [4488, 1971,  513,  ..., 2921, 3452, 2805]])    
        
        tgt_data[:, :-1]
        tensor([[3933, 2734,  325,  ..., 2037, 2672, 2453],
                [1992, 2432,  904,  ..., 1816,  805, 1234],
                [1562, 3574, 3254,  ..., 3248, 3698,  510],
                ...,
                [1542, 4150, 2908,  ..., 1058, 4395,  401],
                [2390, 3349,  358,  ..., 3307, 3117,  710],
                [4488, 1971,  513,  ..., 1952, 2921, 3452]])        
        """
        temp_target_data = tgt_data[:, :-1] # [64, 99]


        #                                  [64, 100]               [64, 99]
        output = transformer(source_data = src_data, target_data = temp_target_data) # [64, 99, 5000] -> [batch_size, max_seq_length - 1, tgt_vocab_size]
        
        #-------------------------------------
        """
        reshapes the output tensor to a 2D tensor -1 tells pytorch to infer the size of the dimension
          from other dimentions, and tgt_vocab_size is the size of the last dimension of the output tensor
          therefore the output tensor is shaped (batch_size * (max_seq_length - 1), tgt_vocab_size)
          [64, 99, 5000] -> [64*99, 5000] -> [6336, 5000]
        note: the contiguous() method allocates a contiguous memory region, which can help avoid potential 
          issues with subsequent operations
        """
        temp_input_data = output.contiguous().view(-1, tgt_vocab_size) # [6336, 5000]
        
        # select all rows, and all columns except the first one [idx 0], then reshape the 2D  tensor into
        #   1D tensor, where .view(-1) means to infer the size of the dimension from the other dimensions
        #            tgt_data = [64,99]      
        temp_target_data = tgt_data[:, 1:].contiguous().view(-1) # [6336]


        loss = criterion(
            input  = temp_input_data, # [6336, 5000]
            target = temp_target_data # [6336]
        )
        #-------------------------------------
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
    #------------------------------

    return {
        'criterion': criterion
    }
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part2(transformer, src_data, tgt_data, tgt_vocab_size):
    result_training = training(
        transformer     = transformer, 
        src_data        = src_data,       # [64,100]
        tgt_data        = tgt_data,       # [64,100]
        tgt_vocab_size  = tgt_vocab_size  # 5000
    )

    return {
        'criterion': result_training['criterion']
    }
#-------------------------------------------------------------------------
result_part2 = execute_part2(
    transformer     = result_part1['transformer'], 
    src_data        = result_part1['src_data'],       # [64,100]
    tgt_data        = result_part1['tgt_data'],       # [64,100]
    tgt_vocab_size  = result_part1['tgt_vocab_size']  # 5000
)
##########################################################################
##
##  PART 3
##
##########################################################################
#-------------------------------------------------------------------------
def model_performance_evaluation(transformer, src_vocab_size, tgt_vocab_size, max_seq_length, criterion):
    """
    This code snippet evaluates the transformer model on a randomly generated validation dataset, 
    computes the validation loss, and prints it. In a real-world scenario, the random validation 
    data should be replaced with actual validation data from the task you are working on. The 
    validation loss can give you an indication of how well your model is performing on unseen 
    data, which is a critical measure of the model's generalization ability
    """

    print('at model_performance_evaluation')
    # src_vocab_size: 5000
    # tgt_vocab_size: 5000
    # max_seq_length: 100

    batch_size = 64

    # Puts the transformer model in evaluation mode. This is important because it turns off certain 
    # behaviors like dropout that are only used during training
    transformer.eval()

    # Generate random sample validation data
    # Random integers between 1 and src_vocab_size, representing a batch of validation source sequences 
    #   with shape (64, max_seq_length)
    val_src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length))  # (batch_size, seq_length)

    # Random integers between 1 and tgt_vocab_size, representing a batch of validation target sequences 
    #   with shape (64, max_seq_length)
    val_tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length))  # (batch_size, seq_length)

    # Disables gradient computation, as we don't need to compute gradients during validation. This can reduce memory 
    #   consumption and speed up computations.
    with torch.no_grad():

        temp_val_tgt_data = val_tgt_data[:, :-1] # [64, 99]
        # Passes the validation source data and the validation target data (excluding the last token in each sequence) 
        #   through the transformer
        #                                     [64, 100]                    [64, 99]
        val_output = transformer(source_data = val_src_data, target_data = temp_val_tgt_data) # [64, 99, 5000]


        temp_input_data = val_output.contiguous().view(-1, tgt_vocab_size) # [6336, 5000]

        # consider that val_tgt_data is [64, 99], therefore 64 * 99 = 6336
        temp_target_data = val_tgt_data[:, 1:].contiguous().view(-1) # [6336]



        # Computes the loss between the model's predictions and the validation target data (excluding the first token 
        #   in each sequence). The loss is calculated by reshaping the data into one-dimensional tensors and using 
        #   the previously defined cross-entropy loss function
        val_loss = criterion(
            input  = temp_input_data, # [6336, 5000]
            target = temp_target_data # [6336]
        )



        print(f"Validation Loss: {val_loss.item()}")
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part3(transformer, src_vocab_size, tgt_vocab_size, max_seq_length, criterion):
    model_performance_evaluation(
        transformer     = transformer,
        src_vocab_size  = src_vocab_size,  # 5000
        tgt_vocab_size  = tgt_vocab_size,  # 5000
        max_seq_length  = max_seq_length,  # 100
        criterion       = criterion
    )
#-------------------------------------------------------------------------   
execute_part3(
    transformer     = result_part1['transformer'], 
    src_vocab_size  = result_part1['src_vocab_size'],  # 5000
    tgt_vocab_size  = result_part1['tgt_vocab_size'],  # 5000
    max_seq_length  = result_part1['max_seq_length'],  # 100
    criterion       = result_part2['criterion']
)
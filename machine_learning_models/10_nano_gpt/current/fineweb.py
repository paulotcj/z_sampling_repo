"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm - progress bar
from contextlib import nullcontext


#-------------------------------------------------------------------------
def setup_os_related():
    identify_os = None
    if os.name == 'nt':
        identify_os = 'windows'
    elif os.name == 'posix':
        if 'darwin' in os.uname().sysname.lower():
            identify_os = 'macos'
        else:
            identify_os = 'linux'
    # print(f'Operating System: {identify_os}')

    if identify_os != 'windows':
        mp.set_start_method('fork') # macOS needs this. Also if this is done later, it will raise an error

    return identify_os
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens

    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))

    tokens_np = np.array(tokens)

    # 2**16 = 65536 tokens
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    
    tokens_np_uint16 = tokens_np.astype(np.uint16)

    return tokens_np_uint16
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def write_datafile(full_file_path, tokens_np):
    np.save(full_file_path, tokens_np) # save the numpy array to a file (.npy)
#-------------------------------------------------------------------------
'''
Note: I know there is code duplication and better ways to do this, but generally speaking, it
separating 2 virtually identical functions for multi and single processing makes it easier to debug,
the code is self-contained in a single location, and multi-processing is quite finicky especially if
you are trying to make it work in multiple platforms. So I intent to keep it as vanilla and original as possible.
'''
#-------------------------------------------------------------------------
def multi_process():
    #-------------------------------------------------------------------------
    with mp.Pool(nprocs) as pool:
        '''
        these vars must be defined here, otherwise, they will be shared between processes
        '''
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((max_shard_size,), dtype=np.uint16) # shard size - 100_000_000

        token_count = 0
        progress_bar = None

        #---------------------------------------------------
        for tokens in pool.imap(tokenize, downloaded_dataset, chunksize=chunksize): #chunksize  = 16

            #----------------
            # is there enough space in the current shard for the new tokens?
            token_count_plus_len_tokens = token_count + len(tokens)
            if token_count_plus_len_tokens < max_shard_size:

                # append tokens to current shard - from current token_count to token_count + len(tokens) will receive the new tokens
                all_tokens_np[token_count:token_count_plus_len_tokens] = tokens 
                token_count += len(tokens)

                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=max_shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))

            else: # write the current shard and start a new one
                
                if shard_index == 0:
                    split = "val"
                else:
                    split = "train"

                filename = f"{file_names_prefix}_{split}_{shard_index:06d}"
                full_file_path = os.path.join(DATA_CACHE_DIR, filename)

                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder_in_shard = max_shard_size - token_count
                progress_bar.update(remainder_in_shard)


                tokens_fit_in_shard = tokens[:remainder_in_shard]
                tokens_remainder = tokens[remainder_in_shard:]

                # write from the current pointer (token_count), to current pointer + remainder_in_shard
                #   use the tokens from 0 to remainder_in_shard
                all_tokens_np[token_count:token_count+remainder_in_shard] = tokens_fit_in_shard
                write_datafile(full_file_path, all_tokens_np)


                shard_index += 1
                progress_bar = None
                
                # populate the next shard with the leftovers of the current doc
                len_tokens_remainder = len(tokens_remainder)
                all_tokens_np[0:len_tokens_remainder] = tokens_remainder
                token_count = len_tokens_remainder
            #----------------
        # end of for loop
        #---------------------------------------------------
    # end of with block
    #---------------------------------------------------

    #---------------------------------------------------
    # write any remaining tokens as the last shard
    if token_count != 0:

        if shard_index == 0:
            split = "val"
        else:
            split = "train"

        '''
        Note that this is the leftover from the first IF block in the for loop. Therefore,
          we have an execat token_count and the tokens were already written to the all_tokens_np
          so all we gotta do is write all_tokens_np from idx 0 to token_count, since the rest is 
          garbage initialized by np.empty((max_shard_size,), dtype=np.uint16)
        '''

        filename = f"{file_names_prefix}_{split}_{shard_index:06d}"
        full_file_path = os.path.join(DATA_CACHE_DIR, filename)

        write_datafile(full_file_path, all_tokens_np[:token_count])  
    #---------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def single_process():
    #-------------------------------------------------------------------------
    with nullcontext():

        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((max_shard_size,), dtype=np.uint16) # shard size - 100_000_000

        token_count = 0
        progress_bar = None

        #---------------------------------------------------
        for sample in downloaded_dataset:
            tokens = tokenize(sample)

            #----------------
            # is there enough space in the current shard for the new tokens?
            token_count_plus_len_tokens = token_count + len(tokens)
            if token_count_plus_len_tokens < max_shard_size:

                # append tokens to current shard - from current token_count to token_count + len(tokens) will receive the new tokens
                all_tokens_np[token_count:token_count_plus_len_tokens] = tokens 
                token_count += len(tokens)

                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=max_shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))

            else: # write the current shard and start a new one
                
                if shard_index == 0:
                    split = "val"
                else:
                    split = "train"

                filename = f"{file_names_prefix}_{split}_{shard_index:06d}"
                full_file_path = os.path.join(DATA_CACHE_DIR, filename)

                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder_in_shard = max_shard_size - token_count
                progress_bar.update(remainder_in_shard)


                tokens_fit_in_shard = tokens[:remainder_in_shard]
                tokens_remainder = tokens[remainder_in_shard:]

                # write from the current pointer (token_count), to current pointer + remainder_in_shard
                #   use the tokens from 0 to remainder_in_shard
                all_tokens_np[token_count:token_count+remainder_in_shard] = tokens_fit_in_shard
                write_datafile(full_file_path, all_tokens_np)


                shard_index += 1
                progress_bar = None
                
                # populate the next shard with the leftovers of the current doc
                len_tokens_remainder = len(tokens_remainder)
                all_tokens_np[0:len_tokens_remainder] = tokens_remainder
                token_count = len_tokens_remainder
            #----------------
        # end of for loop
        #---------------------------------------------------
    # end of with block
    #---------------------------------------------------

    #---------------------------------------------------
    # write any remaining tokens as the last shard
    if token_count != 0:

        if shard_index == 0:
            split = "val"
        else:
            split = "train"

        '''
        Note that this is the leftover from the first IF block in the for loop. Therefore,
          we have an execat token_count and the tokens were already written to the all_tokens_np
          so all we gotta do is write all_tokens_np from idx 0 to token_count, since the rest is 
          garbage initialized by np.empty((max_shard_size,), dtype=np.uint16)
        '''

        filename = f"{file_names_prefix}_{split}_{shard_index:06d}"
        full_file_path = os.path.join(DATA_CACHE_DIR, filename)

        write_datafile(full_file_path, all_tokens_np[:token_count])  
    #---------------------------------------------------
#-------------------------------------------------------------------------


print('\n\n')
print('-------------------------------------------------------------------------')

cpu_count = os.cpu_count()
nprocs = max(1, cpu_count//2)

print(f'cpu_count: {cpu_count}')
print(f'nprocs: {nprocs}')

identify_os = setup_os_related()


print('-------------------------------------------------------------------------')
max_shard_size = int(100_000_000)
# max_shard_size = int(90_000)

#------------
local_dir = "edu_fineweb10B"
file_names_prefix = "edufineweb"
remote_name = "sample-10BT"

dataset_path = "HuggingFaceFW/fineweb-edu"
dataset_split = "train"
#------------


print(f'local_dir: {local_dir}')
print(f'remote_name: {remote_name}')
print(f'shard_size: {max_shard_size}')

print(f'    path: {dataset_path}\n    name: {remote_name}\n    split:{dataset_split}')


print('-------------------------------------------------------------------------')
print('init the tokenizer')
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token


print('-------------------------------------------------------------------------')
# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
print(f'Creating DATA_CACHE_DIR: {DATA_CACHE_DIR}')
os.makedirs(DATA_CACHE_DIR, exist_ok=True) # create DATA_CACHE_DIR, if it exists no error should be raised


print('-------------------------------------------------------------------------')
print('download the dataset')
downloaded_dataset = load_dataset(path = dataset_path, name=remote_name, split=dataset_split)
# downloaded_dataset has 9672101 json entries with documents

print('-------------------------------------------------------------------------')

chunksize = 16
print(f'chunksize:{chunksize}') # for multi-processing



print('-------------------------------------------------------------------------')
print('Starting the process')

# if identify_os == 'windows':
#     ps_ds.single_process()
# else:
#     ps_ds.multi_process()

single_process()
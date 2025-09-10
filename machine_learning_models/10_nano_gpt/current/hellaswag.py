"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{
   "ind":24,
   "activity_label":"Roof shingle removal",
   "ctx_a":"A man is sitting on a roof.",
   "ctx_b":"he",
   "ctx":"A man is sitting on a roof. he",
   "split":"val",
   "split_type":"indomain",
   "label":3,
   "endings":[
      "is using wrap to wrap a pair of skis.",
      "is ripping level tiles off.",
      "is holding a rubik's cube.",
      "starts pulling up roofing on a roof."
   ],
   "source_id":"activitynet~v_-JhWjGDPHMY"
}

ind: dataset ID

activity_label: The ActivityNet or WikiHow label for this example

context: There are two formats. 
    * The full context is in ctx. 
    When the context ends in an (incomplete) noun phrase, like for ActivityNet, this 
    * incomplete noun phrase is in ctx_b, and the 
    * context up until then is in ctx_a. 
    This can be useful for models such as BERT that need the last sentence 
    to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing 
    as ctx_a, followed by a space, then ctx_b.

endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.

split_type: indomain if the activity label is seen during training, else zeroshot

source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

#-------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag") # [...]/.../.../current/hellaswag

#-------------------------------------------------------------------------
def download_file(url: str, fname: str, chunk_size=1024): # this routine is used by 'download' if the file is not already present on disk
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    
    #--------------------------------
    with open(fname, "wb") as file, tqdm( # progress bar / open for writing in binary
        desc=fname,
        total=total,
        unit="iB",         # binary bites
        unit_scale=True,   # scale the unit to a more readable format
        unit_divisor=1024, # 1 KiB = 1024 bytes
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
    #--------------------------------
#-------------------------------------------------------------------------

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val":   "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test":  "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

#-------------------------------------------------------------------------
def download(split): # split can be 'train', 'val', or 'test'
    """Downloads HellaSwag DATA_CACHE_DIR"""

    os.makedirs(DATA_CACHE_DIR, exist_ok=True) # [...]/.../.../current/hellaswag

    data_url = hellaswags[split] # 'train', 'val', or 'test'

    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl") # hellaswag_train.jsonl , hellaswag_val.jsonl, hellaswag_test.jsonl

    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def render_example(example): # this is a json read from typically from iterate_examples(...)
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"] # full sentence context
    label = example["label"] # the correct index answer from the array endings
    endings = example["endings"] # ending sentence options - always 4

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    #--------------------------------
    for end in endings:
        end_tokens = enc.encode(" " + end) # note: prepending " " because GPT-2 tokenizer

        tok_rows.append(ctx_tokens + end_tokens) # the original context plus this option

        # creates a list of zeroes and ones where the number of zeros is equal to the len 
        #   of ctx_tokens, followed by ones where its len is the size of end_tokens
        mask_rows.append( [0]*len(ctx_tokens) + [1]*len(end_tokens) )

        data["ending_tokens"].append(end_tokens)
    #--------------------------------

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max( len(row) for row in tok_rows) # max len of any ctx_tokens + end_tokens

    tokens = torch.zeros((4, max_len), dtype=torch.long) # creates a tensor of zeros shaped [4, max_len]
    mask   = torch.zeros((4, max_len), dtype=torch.long) #  the shape of 4 is due to the 4 ending options

    #--------------------------------
    # zip merges the 2 lists of tok_rows (original context plus options) and mask_rows (list of ones and
    #   zeroes, where the beginning is filled with 0s representing the len of ctx_tokens and the end
    #   filled with 1s representing the len of end_tokens)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):

        # below we are going to create the tensors with the full sentences options (tok_row), and
        #   a mask representing which tokens are the important bits of the options. But they all
        #   have the same length, where the smaller tensors are padded with 0
        tokens[i, :len(tok_row)] = torch.tensor(tok_row) # row i, from idx 0 to len of tok_row
        mask[i, :len(mask_row)] = torch.tensor(mask_row) # row i, from idx 0 to len of masked row

    #--------------------------------

    # data -> the list of options encoded (list of 4 options) with a leading space
    # tokens -> the list of all sentences options. The len is the len of the longest sentence, the excess is padded with zeroes
    # mask -> list of ones and zeroes, where the beginning is filled with 0s representing the len of ctx_tokens and the end filled with 1s representing the len of end_tokens
    # label -> the idx of the correct answer
    return data, tokens, mask, label 
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def iterate_examples(split): # split can be 'train', 'val', or 'test'

    # there are 10,042 examples in total in val
    download(split) # at this point the file should be present on disk

    #--------------------------------
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f: # open file for read
        for line in f:
            example = json.loads(line) #load one line
            yield example # return/yield that line
    #--------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model_type = "gpt2", device = "cuda"):

    torch.set_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type) # gpt2 | from transformers import GPT2LMHeadModel
    model.to(device) # cuda
    # model = torch.compile(model) # optionally torch compile the model

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    #-------------------------------------------------------------------------
    for example in iterate_examples("val"):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        logits = model(tokens).logits
        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()

    # example of usage:
    # python hellaswag.py -m gpt2 -d cuda
    #  or
    # python hellaswag.py --model_type gpt2 --device cuda
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
    parser.add_argument("-d", "--device",     type=str, default="cuda", help="the device to use")

    args = parser.parse_args()
    
    evaluate(args.model_type, args.device)
#-------------------------------------------------------------------------
if __name__ == "__main__":
    main()

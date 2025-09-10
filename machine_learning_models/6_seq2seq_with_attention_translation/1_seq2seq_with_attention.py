##########################################################################
##
##  IMPORTS
##
##########################################################################

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


##########################################################################
##
##  PART 1
##
##########################################################################



#-------------------------------------------------------------------------
class Lang:
    #-------------------------------------------------------------------------
    def __init__(self, name):
        self.name = name
        self.word2index = {} # dictionaries
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"} # first words, SOS: Start Of Sentence, EOS: End Of Sentence
        self.n_words = 2  # Count SOS and EOS
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def addSentence(self, sentence): # get a sentence and try to add a word to the dictionary
        for word in sentence.split(' '):
            self.addWord(word)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words # assume the index of the word dict length, we will add 1 to this count soon
            self.word2count[word] = 1 # word count, first occurrence
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def unicodeToAscii(input_str):

    # unicodedata.normalize('NFD', input_str) -> normalize Unicode strings. Unicode normalization is 
    #   a process that converts text to a standard form, which can be useful for string comparison, 
    #   searching, and other text processing tasks.
    # In this specific call, the function normalize is being used with the normalization form 'NFD'. 
    #   The 'NFD' stands for Normalization Form D (Canonical Decomposition). This form decomposes combined 
    #   characters into their constituent parts. For example, a character like 'é' 
    #   (which is a single character) would be decomposed into 'e' and an accent character.
    #
    # if unicodedata.category(c) != 'Mn' -> In this specific case, the code is checking if the category of 
    #   the character c is not equal to 'Mn'. The category 'Mn' stands for "Mark, Nonspacing". Nonspacing 
    #   marks are characters that typically combine with preceding characters and do not occupy a space 
    #   by themselves, such as accents or diacritical marks.
    #   By using this condition, the code is likely filtering out nonspacing marks from further processing. 
    #   This can be useful in text processing tasks where you want to ignore or remove diacritical marks 
    #   to simplify the text or to perform operations that are sensitive to such marks. For example, 
    #   in text normalization or in preparing text for machine learning models, it might be necessary to 
    #   strip out these marks to ensure consistency and accuracy.    

    return_str = ''.join(
        c for c in unicodedata.normalize('NFD', input_str)
        if unicodedata.category(c) != 'Mn'
    )

    return return_str
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def normalizeString(param_str): # Lowercase, trim, and remove non-letter characters, also separate punctuation from words by addint 1 space

    param_str = unicodeToAscii( param_str.lower().strip() ) # convert to ASCII, lowercase, and strip leading/trailing whitespaces

    # insert a space before every period, exclamation mark, or question mark. ensure that 
    #   punctuation is separated from words by a space, which helpful with text analysis
    param_str = re.sub(r"([.!?])", r" \1", param_str)

    # matches any sequence of one or more characters that are not letters, exclamation marks (!), 
    #   or question marks (?) and replaces them with a single space
    param_str = re.sub(r"[^a-zA-Z!?]+", r" ", param_str)

    return param_str.strip()    
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    file_name = f'data/{lang1}-{lang2}.txt' # most likely: eng-fra.txt
    # Read the file and split into lines
    full_file = open(file_name, encoding='utf-8').read().strip().split('\n')

    # this will be a list of lists, the list len is the number of lines in the file, and the
    #  list[x] length is 2


    sentences_pairs = [
        [
            normalizeString(sentence) 
            for sentence in line.split('\t') # 2 - get each version - the translations are separated by a tab
        ] 
        for line in full_file # 1 - get a line
    ]


    # Reverse pairs, make Lang instances
    if reverse:
        print(f'Reversing the order of the words in the sentences_pairs list. Reverse: {reverse}')
        # reverse the order of the words in the sentences_pairs list
        sentences_pairs = [ list(reversed(pair)) # reverse
                 for pair in sentences_pairs #from each line in the sentences_pairs list
        ]

        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        print(f'Keeping the order of the words in the sentences_pairs list. Reverse: {reverse}')
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

      

    return input_lang, output_lang, sentences_pairs
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def filterPair(pair, lang_prefixes, max_length = 10):

    # filter and filterPairs work together, the intention is to filter out long sentences, typical
    #   max len of 10 words, and also to filter out sentences that do not start with a specific 
    #   prefix - this prefix is a list of typical 'simple' senteces like 'I am sorry'

    # with that in mind, if the length of the first sentence is less than 10, and the 
    #   length of the second sentence is less than 10, and the second sentence starts with

    line_lang1 = pair[0].split(' ')
    len_line_lang1 = len(line_lang1)

    line_lang2 = pair[1].split(' ')
    len_line_lang2 = len(line_lang2)

    return_result = len_line_lang1 < max_length and \
                    len_line_lang2 < max_length and \
                    pair[1].startswith(lang_prefixes)

    return return_result
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def filterPairs(sentences_pairs, lang_prefixes):

    return [
        pair 
        for pair in sentences_pairs 
        if filterPair(pair = pair, lang_prefixes = lang_prefixes) #filter out long sentences and only pick based on the criteria from the prefixes
    ]
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def prepareData(lang1, lang2, lang1_prefixes, reverse=False):

    input_lang, output_lang, sentences_pairs = readLangs(lang1 = lang1, lang2=lang2, reverse = reverse)

    print("Read %s sentence pairs" % len(sentences_pairs))
    

    sentences_pairs = filterPairs(
        sentences_pairs = sentences_pairs, 
        lang_prefixes   = lang1_prefixes
    )


    print("Trimmed to %s sentence pairs" % len(sentences_pairs))

    print("Counting words...")
    for pair in sentences_pairs:
        #remember these are methods from the Lang class, and they count unique words
        input_lang.addSentence(pair[0]) 
        output_lang.addSentence(pair[1])

    print("Counted words:")
    print(f'Language: {input_lang.name} - Number of unique words: {input_lang.n_words}')
    print(f'Language: {output_lang.name} - Number of unique words: {output_lang.n_words}')


    return input_lang, output_lang, sentences_pairs
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part1():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print('Using GPU')
    else:
        print('Using CPU')

    SOS_token = 0
    EOS_token = 1

    MAX_LENGTH = 10

    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )    

    input_lang, output_lang, pairs = prepareData(
        lang1           = 'eng', 
        lang2           = 'fra', 
        lang1_prefixes  = eng_prefixes,  
        reverse         = True
    )
    print(random.choice(pairs))
#-------------------------------------------------------------------------
execute_part1()
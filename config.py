"""
Module: Configuration
Project: Jarvis
Author: Naveen Chakravarthy Balasubramanian
"""

import argparse

"""
Requirements:
    tensorflow = 2.0.1
    tensorflow_datasets = 1.2.0
"""


argp = argparse.ArgumentParser()
argp.add_argument('--max_convos', default=10000, type=int, help='maximum number of conversations to be included')
argp.add_argument('--max_length', default=100, type=int, help='maximum number of words per sentence')
argp.add_argument('--batch_size', default=64, type=int, help='number of conversations to be used per training batch')
argp.add_argument('--num_layers', default=2, type=int, help='number of layers')
argp.add_argument('--num_units', default=512, type=int, help='input dimensions of the final linear layer')
argp.add_argument('--d_model', default=256, type=int, help='dimensions of the model')
argp.add_argument('--num_heads', default=8, type=int, help='number of heads the QKV is to be split into')
argp.add_argument('--dropout', default=0.1, type=float, help='dropout ratio')
argp.add_argument('--activation', default='relu', type=str, help='activation function')
argp.add_argument('--epochs', default=20, type=int, help='number of epochs')
argp.add_argument('--tf_seed', default=1234, type=int, help='random seed for tensorflow operations')
argp.add_argument('--max_vocab_size', default=2**15, type=int, help='maximum size of the vocabulary')
argp.add_argument('--warmup_steps', default=4000, type=int, help='warm up steps for the Learning Rate Schedule')
hyperparams = argp.parse_args()


assert hyperparams.d_model % hyperparams.num_heads == 0

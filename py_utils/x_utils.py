import numpy as np

def pad_to_max_len(seq, max_seq_len, pad=0):
    for i in range(len(seq), max_seq_len):
        seq.append(pad)
    return seq
from text_utils import *
from utils import *
import time
import re
import random
from nltk.tokenize import word_tokenize

def word_position_swap(text, num_swaps=3):
    # Step 1: Identify spans to protect like <EXPRESSION>
    protected_spans = [(m.start(), m.end()) for m in re.finditer(r"<[^<>]+>", text)]

    # Step 2: Tokenize and get character offsets
    tokens = word_tokenize(text)
    offsets = []
    cursor = 0
    for token in tokens:
        start = text.find(token, cursor)
        offsets.append((start, start + len(token)))
        cursor = start + len(token)

    # Step 3: Identify which tokens can be swapped
    swappable_indices = [
        i for i, (start, end) in enumerate(offsets)
        if not any(span_start <= start and end <= span_end for span_start, span_end in protected_spans)
           and re.fullmatch(r'\w+', tokens[i])  # only swap real words
    ]

    # Step 4: Choose random index pairs to swap
    num_pairs = min(num_swaps, len(swappable_indices) // 2)
    indices = random.sample(swappable_indices, 2 * num_pairs)

    # Make sure we work with sorted, non-overlapping pairs
    swap_pairs = list(zip(indices[::2], indices[1::2]))

    # Step 5: Swap in a copy
    swapped = tokens.copy()
    for i, j in swap_pairs:
        swapped[i], swapped[j] = swapped[j], swapped[i]

    result = ' '.join(swapped)
    result  = re.sub(r'<\s*', '<', result)
    result  = re.sub(r'\s*>', '>', result)
    return result


mask_ents('data/base.jsonl', 'data/masked.jsonl')
masked_data = read_jsonl('data/masked.jsonl')
print('>>done masking')

data = [line['text'] for line in masked_data]

a = time.time()
texts = []
for x, sample in enumerate(data):
    sample = sample
    aug = word_position_swap(sample, num_swaps=3)
    texts.append(aug)
    print(x, end=', ')
    # print(aug)
b = time.time()
print(f'total time: {b-a} s')

undo_mask(texts, 1, 'data/masked.jsonl', 'augmented/swap/25-2.jsonl')
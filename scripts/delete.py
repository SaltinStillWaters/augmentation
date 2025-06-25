import time
from text_utils import *
from utils import *

import re
import random
from nltk.tokenize import word_tokenize

def word_deletion_augment(text, deletion_prob=0.2):
    # Step 1: Identify protected spans like <EXPRESSION>
    protected_spans = [(m.start(), m.end()) for m in re.finditer(r"<[^<>]+>", text)]

    # Step 2: Tokenize and map tokens to offsets
    tokens = text.split()
    offsets = []
    cursor = 0
    for token in tokens:
        start = text.find(token, cursor)
        offsets.append((start, start + len(token)))
        cursor = start + len(token)

    # Step 3: Filter tokens for deletion
    new_tokens = []
    for i, (token, (start, end)) in enumerate(zip(tokens, offsets)):
        inside_protected = any(span_start <= start and end <= span_end for span_start, span_end in protected_spans)
        is_word = re.fullmatch(r"\w+", token)
        
        if not inside_protected and is_word:
            if random.random() < deletion_prob:
                continue  # delete this token
        new_tokens.append(token)

    result = ' '.join(new_tokens)
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
    aug = word_deletion_augment(sample, deletion_prob=0.3)
    texts.append(aug)
    print(x, end=', ')
    # print(aug)
b = time.time()
print(f'total time: {b-a} s')

undo_mask(texts, 1, 'data/masked.jsonl', 'augmented/delete/a.jsonl')
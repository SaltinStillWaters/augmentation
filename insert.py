from text_utils import *
from utils import *

from textattack.augmentation.recipes import SynonymInsertionAugmenter
from itertools import chain

augmenter = SynonymInsertionAugmenter(
    pct_words_to_swap=0.2,            
    transformations_per_example=1,
)

# Mask entities first
mask_ents('data/base.jsonl', 'data/masked_noise.jsonl')
masked_data = read_jsonl('data/masked_noise.jsonl')
texts = [line['text'] for line in masked_data]

print('>> done masking')

# Augment
aug_texts = list(chain.from_iterable(augmenter.augment_many(texts)))
print(f'>> done augmenting: {len(aug_texts)} generated')

# Undo mask
undo_mask(aug_texts, 1, 'data/masked_noise.jsonl', 'augmented/delete/25-2.jsonl')  # update filename if needed
print('>> done undoing')

from text_utils import *
from utils import CustomConstraint, mask_ents, undo_mask
from itertools import chain
from textattack.augmentation import Augmenter
from textattack.transformations import (
    WordSwapRandomSwap,
    WordSwapRandomDeletion,
    WordSwapRandomInsertion
)

# Choose one at a time:
transformation = WordSwapRandomSwap()
# transformation = WordSwapRandomDeletion()
# transformation = WordSwapRandomInsertion()

# Optional: add constraints if needed
constraints = [CustomConstraint()]

augmenter = Augmenter(
    transformation=transformation,
    constraints=constraints,
    pct_words_to_swap=0.3,             # percent of words to touch
    transformations_per_example=1,     # generate two variants per input
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
undo_mask(aug_texts, 1, 'data/masked_noise.jsonl', 'augmented/swap/25-1.jsonl')  # update filename if needed
print('>> done undoing')

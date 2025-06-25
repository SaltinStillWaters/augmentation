from text_utils import *
from utils import CustomConstraint, mask_ents, undo_mask

from itertools import chain
from textattack.augmentation import Augmenter
from textattack.transformations import WordSwapEmbedding

# Step 1: Define transformation using static embeddings
transformation = WordSwapEmbedding()

# Step 2: Optionally add your constraint
constraints = [CustomConstraint()]

# Step 3: Create augmenter
augmenter = Augmenter(
    transformation=transformation,
    constraints=constraints,
    pct_words_to_swap=0.3,
    transformations_per_example=1,
)

# Step 4: Mask entities
mask_ents('data/base.jsonl', 'data/masked_2.jsonl')
masked_data = read_jsonl('data/masked_2.jsonl')
texts = [line['text'] for line in masked_data]

print('>> done masking')

# Step 5: Augment
aug_texts = list(chain.from_iterable(augmenter.augment_many(texts)))
print(f'>> done augmenting: {len(aug_texts)} generated')

# Step 6: Undo masks and save
undo_mask(aug_texts, 1, 'data/masked_2.jsonl', 'augmented/semantic/25.jsonl')
print('>> done undoing')

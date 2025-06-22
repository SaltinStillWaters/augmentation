from text_utils import *
from utils import CustomConstraint, mask_ents, undo_mask

from itertools import chain
from textattack.augmentation import Augmenter
from textattack.transformations import BackTranslation

# Step 1: Set up the transformation
transformation = BackTranslation(
    back_translator="textattack/transformer-wmt19-en-de",  # English → German → English
    tokenizer="textattack/transformer-wmt19-de-en"
)

# Optional constraints
constraints = [CustomConstraint()]

# Step 2: Create augmenter
augmenter = Augmenter(
    transformation=transformation,
    constraints=constraints,
    transformations_per_example=1
)

# Step 3: Preprocess
mask_ents('data/base.jsonl', 'data/masked_bt.jsonl')
masked_data = read_jsonl('data/masked_bt.jsonl')
texts = [line['text'] for line in masked_data]

print('>> done masking')

# Step 4: Augment (⚠️ Slower than MLM!)
aug_texts = list(chain.from_iterable(augmenter.augment_many(texts)))
print(f'>> done augmenting: {len(aug_texts)} generated')

# Step 5: Undo mask and save
undo_mask(aug_texts, 1, 'data/masked_bt.jsonl', 'augmented/backtranslation/25.jsonl')
print('>> done undoing')

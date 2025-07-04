from text_utils import *
from utils import CustomConstraint, mask_ents, undo_mask

from itertools import chain
from textattack.augmentation import Augmenter
from textattack.transformations import BackTranslation

# Step 1: Set up the transformation
transformation = BackTranslation()

# Step 2: Create augmenter
augmenter = Augmenter(
    transformation=transformation,
    transformations_per_example=1
)

# Step 3: Preprocess
mask_ents('data/base.jsonl', 'data/masked_bt.jsonl')
masked_data = read_jsonl('data/masked_bt.jsonl')
texts = [line['text'] for line in masked_data]

print('>> done masking')

# Step 4: Augment (⚠️ Slower than MLM!)
aug_texts = []
total = len(texts)
for x, text in enumerate(texts):
    aug_texts.append(augmenter.augment(text)[0])
    print(f'{x}/{total}, ')
print(f'>> done augmenting: {len(aug_texts)} generated')

# Step 5: Undo mask and save
undo_mask(aug_texts, 1, 'data/masked_bt.jsonl', 'augmented/backtranslation/25-2.jsonl')
print('>> done undoing')
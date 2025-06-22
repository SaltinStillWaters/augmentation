import time

from text_utils import *
from utils import CustomConstraint, mask_ents, undo_mask

from itertools import chain
from textattack.augmentation import Augmenter
from textattack.transformations import WordSwapWordNet


transformation = WordSwapWordNet()

constraints = [CustomConstraint()]

augmenter = Augmenter(
    transformation=transformation,
    constraints=constraints,
    pct_words_to_swap=0.3,
    transformations_per_example=1,
)

mask_ents('data/base.jsonl', 'data/masked.jsonl')
masked_data = read_jsonl('data/masked.jsonl')
texts = [line['text'] for line in masked_data]

print('>>done masking')
a = time.time()
aug_texts = list(chain.from_iterable(augmenter.augment_many(texts)))
print(len(aug_texts))
b = time.time()

print('>>done augmenting:', b - a, 's')

undo_mask(aug_texts, 1, 'data/masked.jsonl', 'augmented/wordnet/25.jsonl')
print('>>done undoing')

    
# sentence = "The derivative of the <ENT> is <EXPRESSION> using the chain <ENT>."
# augmented = augmenter.augment(sentence)

# print(augmented)

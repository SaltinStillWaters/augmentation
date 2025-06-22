from text_utils import *
from utils import CustomConstraint, mask_ents, undo_mask

from itertools import chain
from textattack.augmentation import Augmenter
from textattack.transformations import WordSwapMaskedLM
import time
import torch

print(torch.cuda.is_available())  
transformation = WordSwapMaskedLM(
    method="bert-attack"  
)

if torch.cuda.is_available():
    model = transformation.model_wrapper.model
    model.to('cuda')
    model.device = torch.device('cuda')  # Optional but good for consistency
    print(">> Moved MLM model to GPU")
print(">> Model device:", next(transformation.model_wrapper.model.parameters()).device)

constraints = [CustomConstraint()]
augmenter = Augmenter(
    transformation=transformation,
    constraints=constraints,
    pct_words_to_swap=0.3,
    transformations_per_example=1,  
)

mask_ents('data/base.jsonl', 'data/masked_3.jsonl')
masked_data = read_jsonl('data/masked_3.jsonl')
texts = [line['text'] for line in masked_data]

print('>> done masking')
start = time.time()
aug_texts = list(chain.from_iterable(augmenter.augment_many(texts)))
end = time.time()

print(f'>> done augmenting ({len(aug_texts)} results) in {end - start:.2f} s')
undo_mask(aug_texts, 2, 'data/masked_3.jsonl', 'augmented/mlm/25-1.jsonl')
print('>> done undoing')

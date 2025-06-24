import re

from text_utils import *
from textattack.constraints import PreTransformationConstraint
from textattack.transformations import Transformation
from textattack.shared import AttackedText
import random

class WordSwapRandomDeletion(Transformation):
    def __init__(self, pct_deletion, protected_tokens=None):
        super().__init__()
        self.pct_deletion = pct_deletion
        self.protected_tokens = protected_tokens or {"<ENT>", "<EXPRESSION>", "<EQUATION>"}

    def _get_transformations(self, attacked_text, indices_to_modify):
        words = attacked_text.words
        num_words = len(words)

        # Indices eligible for deletion (skip protected tokens)
        eligible_indices = [
            i for i, word in enumerate(words)
            if word not in self.protected_tokens
        ]

        if len(eligible_indices) == 0:
            return []

        num_to_delete = max(1, int(self.pct_deletion * len(words)))
        indices_to_delete = set(random.sample(eligible_indices, min(num_to_delete, len(eligible_indices))))

        new_words = [word for i, word in enumerate(words) if i not in indices_to_delete]

        if not new_words:
            return []

        return [AttackedText(" ".join(new_words))]

    
excludes = ["<", ">", "EXPRESSION", "EQUATION", "ENT"]

class CustomConstraint(PreTransformationConstraint):
    def check_compatibility(self, transformation):
        return True

    def _get_modifiable_indices(self, current_text):
        return set(
            i for i, word in enumerate(current_text.words) if word not in excludes
        )

def mask_ents(jsonl_file, out_file):
    jsonl = read_jsonl(jsonl_file)
    mask = '<ENT>'

    for line in jsonl:
        orig = []
        text = line['text']
        offset = 0
        
        for ent in line['entities']:
            start = ent['start_offset'] + offset
            end = ent['end_offset'] + offset
            
            length = end - start
            offset += len(mask) - length

            orig.append(text[start:end])
            text = f'{text[:start]}{mask}{text[end:]}'

        line['orig'] = orig
        line['text'] = text
        
    save_jsonl(out_file, jsonl)
    
def undo_mask(augmented_sentences, multiplier, masked_file, out_file):
    masked = read_jsonl(masked_file)
    mask = '<ENT>'
    ctr = 0
    new_data = []
    for line in masked:
        for x in range(multiplier):
            text = augmented_sentences[ctr]
            print('->>orig:', text)
            ents = []
            
            for y, orig in enumerate(line['orig']):
                start = text.find(mask)
                end = start + len(mask)
                
                text = f'{text[:start]}{orig}{text[end:]}'

                ents.append({
                    'start_offset': start,
                    'end_offset': start + len(orig),
                    'label': line['entities'][y]['label']
                })

            ctr += 1
            
            new_data.append({
                'text': text,
                'entities': ents,
            })
    
    save_jsonl(out_file, new_data)
            
def check_alignment(file):
    file = read_jsonl(file)
    for line in file:
        for ent in line['entities']:
            print(line['text'][ent['start_offset']:ent['end_offset']])
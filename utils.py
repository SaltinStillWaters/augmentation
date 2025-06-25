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

    for line in jsonl:
        orig = []
        text = line['text']
        offset = 0

        for i, ent in enumerate(line['entities']):
            mask = f"<ENT{i}>"
            start = ent['start_offset'] + offset
            end = ent['end_offset'] + offset

            length = end - start
            offset += len(mask) - length

            orig.append(text[start:end])
            text = f'{text[:start]}{mask}{text[end:]}'

        line['orig'] = orig
        line['text'] = text

    save_jsonl(out_file, jsonl)

    
def undo_mask(aug_texts, num_masks, masked_path, output_path):
    data = read_jsonl(masked_path)
    assert len(aug_texts) == len(data)

    for line, aug in zip(data, aug_texts):
        for i, orig in enumerate(line['orig']):
            aug = re.sub(f"<ENT{i}>", orig, aug)
        line['text'] = aug

    save_jsonl(output_path, data)

            
def check_alignment(file):
    file = read_jsonl(file)
    uniques = set()
    for line in file:
        for ent in line['entities']:
            uniques.add(line['text'][ent['start_offset']:ent['end_offset']])
            # print(line['text'][ent['start_offset']:ent['end_offset']])
    
    return uniques

def fix_ent_order(in_file):
    data = read_jsonl(in_file)
    for line in data:
        line["entities"] = sorted(line["entities"], key=lambda e: e["start_offset"])
    save_jsonl('new.jsonl', data)

fix_ent_order('data/base.jsonl')

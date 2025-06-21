import re

from text_utils import *
from textattack.constraints import PreTransformationConstraint

__excludes = ["<", ">", "EXPRESSION", "EQUATION", "ENT"]

class CustomConstraint(PreTransformationConstraint):
    def check_compatibility(self, transformation):
        return True

    def _get_modifiable_indices(self, current_text):
        return set(
            i for i, word in enumerate(current_text.words) if word not in __excludes
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
            ents = []
            
            for y, orig in enumerate(line['orig']):
                print(text)
                start = text.find(mask)
                end = start + len(mask)
                
                text = f'{text[:start]}{orig}{text[end:]}'

                ents.append({
                    'start_offset': start,
                    'end_offset': end,
                    'label': line['entities'][y]['label']
                })

            ctr += 1
            
            new_data.append({
                'text': text,
                'entities': ents,
            })
    
    save_jsonl(out_file, new_data)
            
# mask_ents('data/base.jsonl', 'data/masked.jsonl')
data = read_jsonl('data/masked.jsonl')
sents = [line['text'] for line in data]

undo_mask(sents, 1, 'data/masked.jsonl', 'data/redo.jsonl')
import sys
import torch
import re
import random
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForMaskedLM
from text_utils import *

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
model.eval()

def mask_entities(text, entities, token="ENT"):
    orig = []
    offset = 0
    for ent in entities:
        start = ent["start_offset"] + offset
        end = ent["end_offset"] + offset
        orig.append(text[start:end])
        text = text[:start] + token + text[end:]
        offset += len(token) - (end - start)
    print('-->>', orig)
    return text, orig

def mask_random_words(text, mask_prob=0.3):
    spans = [(m.start(), m.end()) for m in re.finditer(r"\bENT\b|<[^<>]+>", text)]
    tokens = word_tokenize(text)
    offsets = []
    cursor = 0
    for tok in tokens:
        start = text.find(tok, cursor)
        end = start + len(tok)
        offsets.append((start, end))
        cursor = end
    new_tokens = tokens[:]
    for i, (start, end) in enumerate(offsets):
        inside_protected = any(s <= start < e for s, e in spans)
        if not inside_protected and re.fullmatch(r"\w+", tokens[i]):
            if random.random() < mask_prob:
                new_tokens[i] = tokenizer.mask_token
                
    result = ' '.join(new_tokens)
    result  = re.sub(r'<\s*', '<', result)
    result  = re.sub(r'\s*>', '>', result)
    return result

def restore_entities_with_offsets(text, orig, token="ENT"):
    updated_entities = []
    result = ""
    cursor = 0
    idx = 0

    while idx < len(text):
        if text[idx:idx+len(token)] == token:
            ent = orig[len(updated_entities)]
            start = len(result)
            result += ent
            end = len(result)
            updated_entities.append({"start_offset": start, "end_offset": end})
            idx += len(token)
        else:
            result += text[idx]
            idx += 1
    return result


def apply_mlm_preserve_offsets(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits

    masked_indices = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    input_ids = inputs.input_ids[0].tolist()

    for idx in masked_indices:
        pred_token_id = torch.topk(logits[0, idx], 2).indices[-1].item()
        input_ids[idx] = pred_token_id

    # Convert tokenized version back to text
    new_text = tokenizer.decode(input_ids, skip_special_tokens=True)

    # NOTE: this only works reliably if entities were NOT masked and their text remained unchanged
    # So you can reuse entity_offsets directly assuming the MLM didn’t alter them

    new_text  = re.sub(r'<\s*', '<', new_text)
    new_text  = re.sub(r'\s*>', '>', new_text)
    new_text  = re.sub(r'\s+\'', '\'', new_text)
    new_text  = re.sub(r'\'\s+', '\'', new_text)
    new_text  = re.sub(r'\s+-', '-', new_text)
    new_text  = re.sub(r'-\s+', '-', new_text)
    return new_text

import re

def recompute_entity_offsets(augmented_text, orig_texts, orig_ents):
    """
    Recompute new offsets by locating the exact substrings in augmented text.
    Assumes that entity values from `orig_texts` still exist in augmented text.
    If duplicates exist, finds them in order.
    """
    updated_entities = []
    cursor = 0  # To avoid repeated matches for repeated words

    for ent_text in orig_texts:
        # Escape for regex
        pattern = re.escape(ent_text)
        print('++', augmented_text[cursor:])
        match = re.search(pattern, augmented_text[cursor:], re.IGNORECASE)
        if match:
            start = cursor + match.start()
            end = cursor + match.end()
            updated_entities.append({
                "start_offset": start,
                "end_offset": end
            })
            cursor = end  # move cursor forward to avoid duplicate matches
        else:
            # Fallback if entity text not found
            updated_entities.append({
                "start_offset": -1,
                "end_offset": -1
            })

    labels = [ent['label'] for ent in orig_ents]
    entities = [
    {
        "label": lbl,
        "start_offset": ent["start_offset"],
        "end_offset": ent["end_offset"]
    }
    for lbl, ent in zip(labels, updated_entities)
    ]
    return entities

data = read_jsonl('data/base.jsonl')
result = []

for x, line in enumerate(data):
    try:
        print(x, end=', ')
        masked_text, originals = mask_entities(line['text'], line['entities'])
        random_masked = mask_random_words(masked_text)
        restored = restore_entities_with_offsets(random_masked, originals)
        augmented_text = apply_mlm_preserve_offsets(restored)
        final_entity_offsets = recompute_entity_offsets(augmented_text, originals, line['entities'])
        
        for ent_text, ent_span in zip(originals, final_entity_offsets):
            recovered = augmented_text[ent_span["start_offset"]:ent_span["end_offset"]]
            if ent_text.strip().lower() != recovered.strip().lower():
                print(f"[Mismatch] Expected: '{ent_text}' | Got: '{recovered}'")
                print('\n', '>'*50)
                print('Original:', line['text'])
                print("Masked:", masked_text)
                print("Random Masked:", random_masked)
                print("Restored:", restored)
                print("Augmented:", augmented_text)
                print(final_entity_offsets)
                print('>'*50, '\n')
                sys.exit()
    except:
        break
    
    result.append({
        'text': augmented_text,
        'entities': final_entity_offsets
    })
    
# print(result)
save_jsonl('augmented/mlm/25-1.jsonl', result)
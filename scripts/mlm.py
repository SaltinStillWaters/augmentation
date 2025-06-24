from utils import *
import time
from text_utils import *
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import random
import nltk
import re
import random
import torch
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Load tokenizer + model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
model.eval()


def mlm_augment(text, pct_mask=0.3, top_k=1):
    # Step 1: Identify <...> spans
    spans = [(m.start(), m.end()) for m in re.finditer(r'<[^<>]+>', text)]

    # Step 2: Tokenize and map tokens to char positions
    tokens = word_tokenize(text)
    token_offsets = []
    cursor = 0
    for tok in tokens:
        start = text.find(tok, cursor)
        token_offsets.append((start, start + len(tok)))
        cursor = start + len(tok)

    # Step 3: Identify which token indices are inside <...>
    protected_indices = set()
    for i, (start, end) in enumerate(token_offsets):
        for span_start, span_end in spans:
            if start >= span_start and end <= span_end:
                protected_indices.add(i)
                break

    # Step 4: Randomly select tokens not in protected_indices
    available_indices = [i for i in range(len(tokens)) if i not in protected_indices]
    if not available_indices:
        return text  # No augmentable tokens

    num_to_mask = max(1, int(len(tokens) * pct_mask))
    mask_indices = random.sample(available_indices, min(num_to_mask, len(available_indices)))

    # Step 5: Replace selected tokens with [MASK]
    masked_tokens = tokens.copy()
    for i in mask_indices:
        masked_tokens[i] = '[MASK]'
    masked_text = ' '.join(masked_tokens)

    # Step 6: Run BERT
    inputs = tokenizer(masked_text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits

    # Step 7: Replace [MASK]s with predictions
    masked_token_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    new_tokens = masked_tokens.copy()

    for idx, token_pos in zip(masked_token_indices, mask_indices):
        logits = predictions[0, idx]
        top_predictions = torch.topk(logits, top_k).indices
        replacement = tokenizer.decode([top_predictions[0]]).strip()
        new_tokens[token_pos] = replacement

    result = ' '.join(new_tokens)
    result  = re.sub(r'<\s*', '<', result)
    result  = re.sub(r'\s*>', '>', result)
    return result


mask_ents('data/base.jsonl', 'data/masked.jsonl')
masked_data = read_jsonl('data/masked.jsonl')
print('>>done masking')

data = [line['text'] for line in masked_data]

a = time.time()
texts = []
for x, sample in enumerate(data):
    sample = sample
    aug = mlm_augment(sample, pct_mask=0.3)
    texts.append(aug)
    print(x, end=', ')
    # print(aug)
b = time.time()
print(f'total time: {b-a} s')

undo_mask(texts, 1, 'data/masked.jsonl', 'augmented/mlm/25-2.jsonl')
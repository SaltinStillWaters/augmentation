import torch
import random
import numpy as np
import torchvision

from transformers import EarlyStoppingCallback
from transformers import TrainingArguments, Trainer, AutoModelForTokenClassification
from src.misc.globals import *
from src.pre_process import stage_3
from src.misc.metrics import *


base_out_path = 'out/super_out_6_aug/' # must be suffixed with '/'
dataset_path = 'reannotated_revised_processed/3'

def main(aug_dataset):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(torch.version.cuda)
    print('Torch version:', torch.__version__)
    print('TorchVision version:', torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    

    dataset = stage_3.load(dataset_path)

    best_args = TrainingArguments(
        output_dir=f"{base_out_path}/",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{base_out_path}/logs",
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="tensorboard",
        
        per_device_eval_batch_size=32,
        
        num_train_epochs=3,
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        weight_decay=0.421288,
        warmup_ratio=0.131083,
        fp16=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            id2label=index2tag,
            label2id=tag2index
        )
    
    best_trainer = Trainer(
        args=best_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics_span,
    )

    best_trainer.train()

    try:
        import json
        with open(f'{base_out_path}/results.txt', 'w', encoding='utf-8') as f:
            f.write("best args:\n")
            f.write(json.dumps(best_args.to_dict(), indent=4))
        print(f"best args:\n{best_args}")
    except:
        pass
    
    compute_confusion_matrix(best_trainer, dataset['validation'], labels, f'{base_out_path}summary/matrix_eval.png')

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Optional, but avoids bugs on frozen apps or weird IDEs
    main()

# BioBERT Distillation

[BioBERT pretrained on BC2GM (--teacher_checkpoint)](https://drive.google.com/file/d/1MvXOGpR7JN3iAh2NO1UAu1iHGGkY6358/view?usp=sharing)

```
python distillation.py --task_name ner --model_name bilstm  --path_to_train /bc2gm/train_aug.tsv --path_to_val /bc2gm/val.tsv --model_name_or_path dmis-lab/biobert-base-cased-v1.1 --batch_size 32 --lr_params 1e-3  --num_train_epochs 50 --eval_steps 50 --logging_steps 10 --save_steps 10  --weight_decay 1e-2 --teacher_checkpoint biobert_state_dict.pt --do_train 1 --distillation 1
```

### Data augmentation

Done as in [Dai and Adel, 2020](https://arxiv.org/pdf/2010.11683.pdf) except for the synonim replacement. Methods used are:
* Label-wise replacement (with probability *p* substitute word with another one that has the same IOBES label)
* Mention replacement (with probability *p* substitute gene mention with another gene mention that might be of a different label)
* Shuffle within segments (with probability *p* select segments of the same type (genes and O-tokens) and shuffle).

All augmentation approaches use *p=0.5*.

### Results
*W/O distillation*

NE count train** | Learning rate  | BERT embeddings | Embedding size | LSTM hidden size | Classifier hidden size* | Epochs | F1 score | Size
----- | ------------ | ------------- | ------------ | ------------- | ------------ | ------------ | -------- | ------
15K | 1e-3 | - | 300 | 300 | - | 50 | 0.7745 | 47M
15K | 1e-3 | - | 300 | 300 | 256 | 50 | 0.7742 | 47.6M
262K | 1e-3 | - | 300 | 300 | 256 | 35 | 0.7786 | 47.6M

*With distillation*

NE count train** | Learning rate  | BERT embeddings | Embedding size | LSTM hidden size | Classifier hidden size* | Epochs | F1 score | Size
----- | ------------ | ------------- | ------------ | ------------- | ------------ | ------------ | -------- | ------
15K | 1e-3 | - | 300 | 300 | 256 | 50 | 0.7741 | 46.9M
56K | 1e-3 | - | 300 | 300 | - | 50 | 0.8004 | 46.9M
56K | 1e-3 | - | 300 | 300 | 256 | 50 | 0.8010 | 47.6M
56K | 1e-3 | + | 768 | 300 | 256 | 30 | 0.8130 | 105M
138K | 1e-3 | - | 300 | 200 | 256 | 50 | 0.8165 | 40.3M
138K | 1e-3 | - | 300 | 300| 256 | 50 | 0.8210 | 47.6M
262K | 1e-3 | - | 300 | 300 | 256 | 30 | 0.8284 | 47.6M


&ast; classifier hidden size '-' means one linear layer was used

&ast;&ast; number of gene mentions in the dataset (15K -- original, other numbers -- augmented)
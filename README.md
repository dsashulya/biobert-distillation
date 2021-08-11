# BioBERT Distillation

[BioBERT pretrained on BC2GM]()

```
python distillation.py --task_name ner --model_name bilstm  --path_to_train /bc2gm/train_aug.tsv --path_to_val /bc2gm/val.tsv --model_name_or_path dmis-lab/biobert-base-cased-v1.1 --batch_size 32 --lr_params 1e-3  --num_train_epochs 50 --eval_steps 50 --logging_steps 10 --save_steps 10  --weight_decay 1e-2 --teacher_checkpoint biobert_state_dict.pt --do_train 1 --distillation 1
```

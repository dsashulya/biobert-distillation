import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch import nn
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    AdamW, BertForTokenClassification, BertConfig
)

from data import get_bc2gm_train_data
from data import get_ner_model_inputs
from eval import evaluate_ner_metrics
from log import setup_logging
from loss import loss as ner_loss
from ner_utils import build_dict
from tags import UTIL_TAGS
from train import set_seed


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=10, type=int, required=False)
    parser.add_argument('--do_train', default=False, type=lambda x: bool(int(x)), required=False)
    parser.add_argument('--do_eval', default=False, type=lambda x: bool(int(x)), required=False)
    parser.add_argument('--local_rank', default=-1, type=int, required=False)
    parser.add_argument('--world_size', default=1, type=int, required=False)
    parser.add_argument('--n_gpu', default=1, type=int, required=False)
    parser.add_argument('--logging_level', default=20, type=int, required=False)
    parser.add_argument('--model_name_or_path', default=None, type=str, required=True,
                        help="used in model_class.from_pretrained()")
    parser.add_argument('--teacher_checkpoint', default=None, type=str, required=False,
                        help="checkpoint to load the model from")
    parser.add_argument('--student_checkpoint', default=None, type=str, required=False,
                        help="checkpoint to load the model from")

    # data params
    parser.add_argument('--path_to_train', default=None, type=str, required=False)
    parser.add_argument('--path_to_val', default=None, type=str, required=False)
    parser.add_argument('--batch_size', default=16, type=int, required=False)
    parser.add_argument('--tokenizer_name', default=None, type=str, required=False)
    parser.add_argument('--use_fast', default=True, type=lambda x: bool(int(x)), required=False,
                        help="whether to use fast tokenizer")

    # model params
    parser.add_argument("--task_name", default=None, type=str, required=True)
    parser.add_argument('--model_name', default=None, type=str, required=True)
    parser.add_argument('--weight_decay', default=0., type=float, required=False)
    parser.add_argument('--lr_params', default=5e-5, type=float, required=False)
    parser.add_argument('--scheduler', default='const', type=str, required=False)
    parser.add_argument('--warmup_steps', default=0, type=int, required=False)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, required=False)
    parser.add_argument('--max_grad_norm', default=1., type=float, required=False)
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

    # train params
    parser.add_argument('--num_train_epochs', default=5, type=int, required=False)
    parser.add_argument('--distillation', default=False, type=lambda x: bool(int(x)), required=True)
    parser.add_argument('--alpha', default=0., type=float, required=False)
    parser.add_argument('--logging_steps', default=5, type=int, required=False)
    parser.add_argument('--eval_steps', default=5, type=int, required=False)
    parser.add_argument('--write', default=True, type=lambda x: bool(int(x)), required=False,
                        help="Write logs to summary writer")
    parser.add_argument('--save_steps', type=int, default=10,
                        help="Save last checkpoint every X update steps")
    parser.add_argument('--update_steps_start', type=int, default=0,
                        help="when using pretrained model enter how many update steps it already underwent")
    return parser


@dataclass
class BiLSTMConfig:
    n_layers: int
    embedding_size: int
    hidden_size: int
    dropout: float


@dataclass
class BiLSTMOutput:
    logits: torch.Tensor


class BiLSTMForTokenClassification(nn.Module):
    def __init__(self, config: BiLSTMConfig, vocab_size: int, n_classes: int, device: torch.device):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, config.embedding_size)
        self.lstm = nn.LSTM(input_size=config.embedding_size,
                            hidden_size=config.hidden_size,
                            num_layers=config.n_layers,
                            dropout=config.dropout,
                            batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(2 * config.hidden_size, n_classes)
        self.device = device

    def forward(self, input_ids: torch.Tensor, **kwargs):
        emb = self.embedding(input_ids)
        out, (_, _) = self.lstm(emb)
        out = self.linear(out)
        return BiLSTMOutput(logits=out)


def evaluate_and_write(model, writer, eval_func, val_dataloader, update_steps, **kwargs):
    model.eval()
    val_loss, metrics = eval_func(model, val_dataloader, **kwargs)
    if writer is not None:
        writer.add_scalar('Losses/val', val_loss, update_steps)
        for name, metric in metrics.items():
            writer.add_scalar(f'Metrics/{name}_dev', metric, update_steps)
    return val_loss, metrics


def write_params(writer, args):
    output = f'Model path {args.model_name_or_path}  \n'
    output += f'Task name {args.task_name}  \n'
    output += f'Use distillation {args.distillation}  \n'
    output += f'Epochs {args.num_train_epochs}  \n'
    output += f'Learning rate {args.lr_params}  \n'
    output += f'Weight decay {args.weight_decay}  \n'
    output += f'Batch size {args.batch_size}  \n'
    output += f'Distillation alpha {args.alpha}  \n'
    output += f'Start update steps {args.update_steps_start}'
    writer.add_text('Parameters', output, args.update_steps_start)


def train(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    saving_name = f'{args.model_name}-{datetime.now():%Y%m%d-%H%M-%S}'
    # preparing summary writer

    label2id = build_dict(UTIL_TAGS + ['GENE'], ['B-', 'I-', 'E-', 'S-'])
    label_map = {value: key for key, value in label2id.items()}
    num_labels = len(label_map)

    config = BiLSTMConfig(
        n_layers=2,
        embedding_size=300,
        hidden_size=300,
        dropout=0.5
    )
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BiLSTMForTokenClassification(config, tokenizer.vocab_size, num_labels, device).to(device)
    if args.student_checkpoint is not None:
        model.load_state_dict(torch.load(args.student_checkpoint))

    if args.distillation:
        teacher_config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            id2label=label_map,
            label2id=label2id,
        )
        teacher_model = BertForTokenClassification.from_pretrained(args.model_name_or_path, config=teacher_config).to(
            device)
        if args.teacher_checkpoint is not None:
            teacher_model.load_state_dict(torch.load(args.teacher_checkpoint))
        teacher_model.eval()

    optimizer = AdamW(model.parameters(), lr=args.lr_params, weight_decay=args.weight_decay)

    train_dataloader, val_dataloader = get_bc2gm_train_data(args, tokenizer,
                                                            label2id, return_train=True, return_val=True)

    update_steps = args.update_steps_start
    best_f1 = -np.inf
    set_seed(args)

    writer = None
    if args.write:
        writer = setup_logging(args.task_name + '_' + args.model_name + ('_distil' if args.distillation else ''))
        write_params(writer, args)
    # eval
    _, metrics = evaluate_and_write(model, writer, evaluate_ner_metrics, val_dataloader, update_steps,
                                    label_map=label_map, tokenizer=tokenizer)
    for epoch in range(args.num_train_epochs):
        epoch_iterator = tqdm(train_dataloader, desc="Train iteration", position=0, leave=True)
        for step, batch in enumerate(epoch_iterator):
            model.train()

            tags_vocab = {value: key for key, value in label_map.items()}
            batch = get_ner_model_inputs(batch, tokenizer, tags_vocab)
            output = model(**{key: value.to(model.device) for key, value in batch.items()})

            if args.distillation:
                with torch.no_grad():
                    teacher_output = teacher_model(**{key: value.to(model.device) for key, value in batch.items()})
                loss_bilstm = ner_loss(output.logits.cpu(), batch['labels'], batch['attention_mask'])
                loss_distill = F.mse_loss(teacher_output.logits, output.logits)
                loss = args.alpha * loss_bilstm + (1 - args.alpha) * loss_distill
            else:
                loss = ner_loss(output.logits.cpu(), batch['labels'], batch['attention_mask'])

            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps

            loss.backward()
            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                update_steps += 1

                if writer is not None:
                    writer.add_scalar('Losses/train',
                                      loss.item(), update_steps)

                if update_steps % args.eval_steps == 0:
                    _, metrics = evaluate_and_write(model, writer, evaluate_ner_metrics, val_dataloader, update_steps,
                                                    label_map=label_map, tokenizer=tokenizer)
                    if metrics['f1'] > best_f1:
                        torch.save(model.state_dict(), f'{saving_name}_best.pt')
                        best_f1 = metrics['f1']

                if update_steps % args.save_steps:
                    torch.save(model.state_dict(), f'{saving_name}_last.pt')


def eval(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    label2id = build_dict(UTIL_TAGS + ['GENE'], ['B-', 'I-', 'E-', 'S-'])
    label_map = {value: key for key, value in label2id.items()}
    num_labels = len(label_map)
    teacher_config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id=label2id,
    )
    teacher_model = BertForTokenClassification.from_pretrained(args.model_name_or_path, config=teacher_config).to(
        device)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    if args.teacher_checkpoint is not None:
        teacher_model.load_state_dict(torch.load(args.teacher_checkpoint))
    teacher_model.eval()

    train_dataloader, val_dataloader = get_bc2gm_train_data(args, tokenizer,
                                                            label2id, return_train=True, return_val=True)

    _, met = evaluate_ner_metrics(teacher_model, val_dataloader, label_map, tokenizer)
    print(met)


if __name__ == "__main__":
    args = setup_argparser().parse_args()
    if args.do_train:
        train(args)
    else:
        eval(args)

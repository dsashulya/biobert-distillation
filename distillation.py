import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    AdamW, BertForTokenClassification, BertConfig,
    PreTrainedTokenizer
)

from data import get_bc2gm_train_data
from data import get_ner_model_inputs
from eval import evaluate_ner_metrics
from log import setup_logging
from loss import loss as ner_loss
from ner_utils import build_dict
from tags import UTIL_TAGS
import random
from copy import deepcopy


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


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
    parser.add_argument("--dropout", default=0.5, type=float, help="bilstm dropout")
    parser.add_argument("--n_layers", default=2, type=int, help="bilstm number of layers")
    parser.add_argument("--hidden_size", default=300, type=int, help="bilstm hidden size")
    parser.add_argument("--classifier_size", default=256, type=int, help="bilstm classifier hidden size")
    parser.add_argument('--embedding_type', default='train', type=str,
                        help='embeddings used in bilstm: train (nn.Embedding), bert or word2vec')
    parser.add_argument('--embedding_size', default=300, type=int,
                        help='used when embedding is set to train')

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

    parser.add_argument('--comment', type=str, default=None, help='additional info to log')
    return parser


@dataclass
class BiLSTMConfig:
    n_layers: int
    embedding_size: int
    hidden_size: int
    dropout: float
    classifier_size: int


@dataclass
class BiLSTMOutput:
    logits: torch.Tensor
    

class MultiChannelEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, out_channels, filters: list):
        # filters must each be an odd number otherwise token number is lost
        # (might be that convolutions work only for sequence classification)
        super().__init__()
        self.filters_size = out_channels
        self.filters = filters

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.conv = nn.ModuleList([
            nn.Conv1d(embedding_size, out_channels, kernel_size=filter, padding=filter // 2)
            for filter in filters
        ])

    def init_embedding(self, weight):
        self.embedding.weight = nn.Parameter(weight.to(self.embedding.weight.device))

    def forward(self, input_ids, **kwargs):
        emb = self.embedding(input_ids).transpose(1, 2)
        filters = []
        for conv1d in self.conv:
            filters.append(conv1d(emb).transpose(1, 2))
        out = F.relu(torch.cat(filters, dim=2))
        return out


class BiLSTMForTokenClassification(nn.Module):
    def __init__(self, config: BiLSTMConfig, vocab_size: int, n_classes: int, device: torch.device, embedding,
                 bert=None):
        super().__init__()

        self.embedding_type = embedding
        if embedding == 'bert' and bert is not None:
            self.embedding = deepcopy(bert.bert.embeddings)

        elif embedding == 'multichannel':
            self.embedding = MultiChannelEmbedding(vocab_size=vocab_size,
                                                   embedding_size=config.embedding_size,
                                                   out_channels=256,
                                                   filters=[1, 3, 5])
            config.embedding_size = len(self.embedding.filters) * self.embedding.filters_size
        else:
            self.embedding = nn.Embedding(vocab_size, config.embedding_size)

        self.lstm = nn.LSTM(input_size=config.embedding_size,
                            hidden_size=config.hidden_size,
                            num_layers=config.n_layers,
                            dropout=config.dropout,
                            batch_first=True,
                            bidirectional=True)
        if not config.classifier_size:
            self.linear = nn.Linear(2 * config.hidden_size, n_classes)
        else:
            self.linear = nn.Sequential(
                nn.Linear(2 * config.hidden_size, config.classifier_size),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.classifier_size, n_classes)
            )
        self.dropout = nn.Dropout(0.5)
        self.device = device

    def forward(self, input_ids: torch.Tensor, **kwargs):
        emb = self.embedding(input_ids)
        out, _ = self.lstm(emb)
        out = self.linear(self.dropout(out))
        return BiLSTMOutput(logits=out)

    def forward_(self, input_ids: torch.Tensor, **kwargs):
        emb = self.embedding(input_ids)
        out, (_, _) = self.lstm(emb)
        out = self.linear(out)
        return BiLSTMOutput(logits=out)


def write_params(writer, args):
    output = f'Model path {args.model_name_or_path}  \n'
    output += f'Task name {args.task_name}  \n'
    output += f'Use distillation {args.distillation}  \n'
    output += f'Epochs {args.num_train_epochs}  \n'
    output += f'Learning rate {args.lr_params}  \n'
    output += f'Weight decay {args.weight_decay}  \n'
    output += f'Batch size {args.batch_size}  \n'
    output += f'Distillation alpha {args.alpha}  \n'
    output += f'BiLSTM number of layers {args.n_layers}  \n'
    output += f'BiLSTM hidden size {args.hidden_size}  \n'
    output += f'BiLSTM embedding type {args.embedding_type}  \n'
    output += f'BiLSTM embedding size {args.embedding_size}  \n'
    output += f'BiLSTM classifier size {args.classifier_size}  \n'
    output += f'BiLSTM dropout rate {args.dropout}  \n'
    output += f'Start update steps {args.update_steps_start}  \n'
    if args.comment is not None:
        output += args.comment
    writer.add_text('Parameters', output, args.update_steps_start)


def init_model(args):
    if args.embedding_type == 'bert':
        args.embedding_size = 768

    label2id = build_dict(UTIL_TAGS + ['GENE'], ['B-', 'I-', 'E-', 'S-'])
    label_map = {value: key for key, value in label2id.items()}
    num_labels = len(label_map)
    args.label2id = label2id
    args.label_map = label_map

    teacher_model = None
    if args.distillation and (args.do_train or args.embedding_type == 'bert'):
        teacher_config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            id2label=label_map,
            label2id=label2id,
        )
        teacher_model = BertForTokenClassification.from_pretrained(args.model_name_or_path, config=teacher_config).to(
            args.device)
        if args.teacher_checkpoint is not None:
            teacher_model.load_state_dict(torch.load(args.teacher_checkpoint))
        teacher_model.eval()

    config = BiLSTMConfig(
        n_layers=args.n_layers,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        classifier_size=args.classifier_size
    )
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    if args.embedding_type == 'bert' and args.distillation:
        model = BiLSTMForTokenClassification(config, tokenizer.vocab_size, num_labels, args.device,
                                             bert=teacher_model, embedding=args.embedding_type).to(args.device)
    else:
        model = BiLSTMForTokenClassification(config, tokenizer.vocab_size, num_labels, args.device,
                                             bert=None, embedding=args.embedding_type).to(args.device)

    if args.student_checkpoint is not None:
        model.load_state_dict(torch.load(args.student_checkpoint))
    return tokenizer, model, teacher_model


class DistillLoss:
    def __init__(self, loss_func: callable, teacher: nn.Module, alpha: float):
        self.teacher = teacher
        self.alpha = alpha
        self.loss_func = loss_func

    def get_loss(self, logits: torch.Tensor, batch: dict):
        with torch.no_grad():
            teacher_output = self.teacher(**batch)
        loss = self.loss_func(logits.cpu(), batch['labels'].cpu(), batch['attention_mask'].cpu())
        loss_distill = F.mse_loss(teacher_output.logits, logits)
        return args.alpha * loss + (1 - args.alpha) * loss_distill


class NoDistillLoss:
    def __init__(self, loss_func: callable, *args, **kwargs):
        self.loss_func = loss_func

    def get_loss(self, logits: torch.Tensor, batch: dict):
        return self.loss_func(logits.cpu(), batch['labels'].cpu(), batch['attention_mask'].cpu())


class NerData:
    def __init__(self, tokenizer: PreTrainedTokenizer, tags_vocab: dict):
        self.tokenizer = tokenizer
        self.tags_vocab = tags_vocab

    def get_inputs(self, batch):
        return get_ner_model_inputs(batch, self.tokenizer, self.tags_vocab)


class Data:
    def get_inputs(self, batch):
        return batch


class Evaluator:
    def __init__(self, model: nn.Module, eval_func: callable, val_dataloader: DataLoader,
                 writer: SummaryWriter = None, saving_name=None):
        self.model = model
        self.writer = writer
        self.eval_func = eval_func
        self.val_dataloader = val_dataloader
        self.best_f1 = -np.inf
        self.saving_name = saving_name

    def evaluate_and_write(self, update_steps: int, **kwargs):
        self.model.eval()
        val_loss, metrics = self.eval_func(self.model, self.val_dataloader, **kwargs)
        if self.writer is not None:
            self.writer.add_scalar('Losses/val', val_loss, update_steps)
            for name, metric in metrics.items():
                self.writer.add_scalar(f'Metrics/{name}_dev', metric, update_steps)

        if metrics['f1'] > self.best_f1 and self.saving_name is not None:
            torch.save(self.model.state_dict(), f'{self.saving_name}_best.pt')
            self.best_f1 = metrics['f1']
        return val_loss, metrics


def train(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device
    saving_name = f'{args.model_name}-{datetime.now():%Y%m%d-%H%M-%S}'

    tokenizer, model, teacher_model = init_model(args)
    optimizer = AdamW(model.parameters(), lr=args.lr_params, weight_decay=args.weight_decay)

    train_dataloader, val_dataloader = get_bc2gm_train_data(args, tokenizer,
                                                            args.label2id, return_train=True, return_val=True)

    update_steps = args.update_steps_start
    set_seed(args)

    writer = None
    if args.write:
        writer = setup_logging(saving_name)
        write_params(writer, args)

    tags_vocab = {value: key for key, value in args.label_map.items()}

    loss_func = ner_loss if args.task_name == 'ner' else None
    eval_func = evaluate_ner_metrics if args.task_name == 'ner' else None

    loss_cls = DistillLoss(loss_func, teacher_model, args.alpha) if args.distillation else NoDistillLoss(loss_func)
    data_cls = NerData(tokenizer, tags_vocab) if args.task_name == 'ner' else Data()
    evaluator = Evaluator(model, eval_func, val_dataloader, writer, saving_name)
    # eval
    _, metrics = evaluator.evaluate_and_write(update_steps,
                                              label_map=args.label_map, tokenizer=tokenizer)
    for epoch in range(args.num_train_epochs):
        epoch_iterator = tqdm(train_dataloader, desc="Train iteration", position=0, leave=True)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = data_cls.get_inputs(batch)
            batch = {key: value.to(model.device) for key, value in batch.items()}
            output = model(**batch)

            loss = loss_cls.get_loss(output.logits, batch)

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
                    _, metrics = evaluator.evaluate_and_write(update_steps,
                                                              label_map=args.label_map, tokenizer=tokenizer)

                if update_steps % args.save_steps:
                    torch.save(model.state_dict(), f'{saving_name}_last.pt')


def eval(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device

    tokenizer, model, _ = init_model(args)
    model.eval()

    writer = None
    _, val_dataloader = get_bc2gm_train_data(args, tokenizer, args.label2id, return_train=False, return_val=True)
    # eval
    eval_func = evaluate_ner_metrics if args.task_name == 'ner' else None
    evaluator = Evaluator(model, eval_func, val_dataloader, writer)
    _, metrics = evaluator.evaluate_and_write(0,
                                              label_map=args.label_map, tokenizer=tokenizer)

    print(metrics)


if __name__ == "__main__":
    args = setup_argparser().parse_args()
    if args.do_train:
        train(args)
    else:
        eval(args)

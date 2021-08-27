import argparse
from datetime import datetime
from typing import Tuple, Any

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
    parser.add_argument('--measure_time', default=False, type=lambda x: bool(int(x)), required=False)
    parser.add_argument('--do_train', default=False, type=lambda x: bool(int(x)), required=False)
    parser.add_argument('--do_eval', default=False, type=lambda x: bool(int(x)), required=False)
    parser.add_argument('--local_rank', default=-1, type=int, required=False)
    parser.add_argument('--world_size', default=1, type=int, required=False)
    parser.add_argument('--n_gpu', default=1, type=int, required=False)
    parser.add_argument('--logging_level', default=20, type=int, required=False)
    parser.add_argument('--teacher_model_name_or_path', default=None, type=str, required=False,
                        help="used in model_class.from_pretrained()")
    parser.add_argument('--student_model_name_or_path', default=None, type=str, required=False,
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
    output = f'Teacher model path {args.teacher_model_name_or_path}  \n'
    output += f'Student model path {args.student_model_name_or_path}  \n'
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
    teacher_tokenizer = None
    teacher_config = None
    if args.distillation and (args.do_train or args.embedding_type == 'bert' or (args.measure_time
                                                                                 and args.teacher_checkpoint)):
        teacher_config = BertConfig.from_pretrained(
            args.teacher_model_name_or_path,
            num_labels=num_labels,
            id2label=label_map,
            label2id=label2id,
        )
        teacher_model = BertForTokenClassification.from_pretrained(args.teacher_model_name_or_path,
                                                                   config=teacher_config).to(
            args.device)
        if args.teacher_checkpoint is not None:
            teacher_model.load_state_dict(torch.load(args.teacher_checkpoint))
        teacher_model.eval()

        teacher_tokenizer = BertTokenizer.from_pretrained(args.teacher_model_name_or_path)

    if args.model_name.lower() == 'bilstm':
        config = BiLSTMConfig(
            n_layers=args.n_layers,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            classifier_size=args.classifier_size
        )
        student_tokenizer = BertTokenizer.from_pretrained(args.teacher_model_name_or_path)
        if args.embedding_type == 'bert' and args.distillation:
            model = BiLSTMForTokenClassification(config, student_tokenizer.vocab_size, num_labels, args.device,
                                                 bert=teacher_model, embedding=args.embedding_type).to(args.device)
        else:
            model = BiLSTMForTokenClassification(config, student_tokenizer.vocab_size, num_labels, args.device,
                                                 bert=None, embedding=args.embedding_type).to(args.device)
    else:
        config = BertConfig(
            attention_probs_dropout_prob=0.1,
            cell={},
            model_type="bert",
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            hidden_size=312,
            initializer_range=0.02,
            intermediate_size=1200,
            max_position_embeddings=512,
            num_attention_heads=12,
            num_hidden_layers=4,
            pre_trained="",
            structure=[],
            type_vocab_size=2,
            vocab_size=28996,
            num_labels=num_labels,
            id2label=label_map,
            label2id=label2id,
        )
        student_tokenizer = BertTokenizer.from_pretrained(args.teacher_model_name_or_path)
        model = TinyBertForTokenClassification(config=config, device=args.device).to(args.device)

    if args.student_checkpoint is not None:
        model.load_state_dict(torch.load(args.student_checkpoint))
    return student_tokenizer, model, teacher_tokenizer, teacher_model


class DistillLoss:
    def __init__(self, loss_func: callable, teacher: nn.Module, alpha: float):
        self.teacher = teacher
        self.alpha = alpha
        self.loss_func = loss_func

    def get_loss(self, output, batch: dict):
        with torch.no_grad():
            teacher_output = self.teacher(**batch, output_hidden_states=True, output_attentions=True)

        loss = self.loss_func(output.logits.cpu(), batch['labels'].cpu(), batch['attention_mask'].cpu())
        loss_distill = F.mse_loss(teacher_output.logits, output.logits)
        if output.attentions and output.hidden_states:
            loss_emb = F.mse_loss(teacher_output.hidden_states[0], output.hidden_states[0])
            hidden_aligned = [teacher_output.hidden_states[i] for i in range(1, len(teacher_output.hidden_states), 3)]
            loss_hid, loss_attn = 0., 0.
            for teacher_hid, student_hid in zip(hidden_aligned, output.hidden_states):
                loss_hid += F.mse_loss(teacher_hid, student_hid)

            attn_aligned = [teacher_output.attentions[i] for i in range(0, len(teacher_output.attentions), 3)]
            for teacher_attn, student_attn in zip(attn_aligned, output.attentions):
                loss_attn += F.mse_loss(teacher_attn, student_attn)
            loss_distill += loss_emb + loss_hid + loss_attn

        return args.alpha * loss + (1 - args.alpha) * loss_distill



class NoDistillLoss:
    def __init__(self, loss_func: callable, *args, **kwargs):
        self.loss_func = loss_func

    def get_loss(self, logits: torch.Tensor, batch: dict):
        return self.loss_func(logits.cpu(), batch['labels'].cpu(), batch['attention_mask'].cpu())


class NerData:
    def __init__(self, student_tokenizer: PreTrainedTokenizer,
                 teacher_tokenizer: PreTrainedTokenizer,
                 tags_vocab: dict):
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.tags_vocab = tags_vocab

    def get_inputs(self, batch, teacher=False):
        return get_ner_model_inputs(batch, self.student_tokenizer if not teacher else self.teacher_tokenizer,
                                    self.tags_vocab)


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


class TinyBertForTokenClassification(nn.Module):
    def __init__(self, config, fit_size=768, device: torch.device = torch.device("cuda")):
        super(TinyBertForTokenClassification, self).__init__()
        self.bert = BertForTokenClassification(config=config)
        self.W_emb = nn.Linear(config.hidden_size, fit_size)
        self.W_hidden = nn.Linear(config.hidden_size, fit_size)
        self.device = device

    def forward(self, input_ids, **kwargs):
        out = self.bert(input_ids, **kwargs, output_hidden_states=True, output_attentions=True)
        logits, hidden, attentions = out.logits, out.hidden_states, out.attentions
        hidden = tuple([self.W_emb(hidden[0])] + [self.W_hidden(hidden_el) for hidden_el in hidden[1:]])
        return TinyBertOutput(logits=logits, hidden_states=hidden, attentions=attentions)


@dataclass
class TinyBertOutput:
    logits: torch.Tensor
    hidden_states: Tuple[Any]
    attentions: Tuple[Any]

@dataclass
class BiLSTMOutput:
    logits: torch.Tensor
    hidden_states: Tuple[Any] = tuple()
    attentions: Tuple[Any] = tuple()


def train(args):
    assert args.model_name.lower() in ['bilstm', 'tinybert'], "Model not supported"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device
    saving_name = f'{args.model_name}-{datetime.now():%Y%m%d-%H%M-%S}'

    student_tokenizer, model, teacher_tokenizer, teacher_model = init_model(args)
    optimizer = AdamW(model.parameters(), lr=args.lr_params, weight_decay=args.weight_decay)

    # student tokenizer used
    train_dataloader, val_dataloader = get_bc2gm_train_data(args, student_tokenizer,
                                                            args.label2id, return_train=True, return_val=True)

    teacher_train_dataloader = [[] for _ in range(len(train_dataloader))]
    if args.student_model_name_or_path:
        # teacher tokenizer used
        teacher_train_dataloader, _ = get_bc2gm_train_data(args, teacher_tokenizer,
                                                           args.label2id, return_train=True, return_val=False)

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
    data_cls = NerData(student_tokenizer, teacher_tokenizer, tags_vocab) if args.task_name == 'ner' else Data()
    evaluator = Evaluator(model, eval_func, val_dataloader, writer, saving_name)
    # eval
    _, metrics = evaluator.evaluate_and_write(update_steps,
                                              label_map=args.label_map, tokenizer=student_tokenizer)
    for epoch in range(args.num_train_epochs):
        epoch_iterator = tqdm(zip(train_dataloader, teacher_train_dataloader),
                              desc="Train iteration", position=0, leave=True, total=len(train_dataloader))
        for step, (batch, teacher_batch) in enumerate(epoch_iterator):
            model.train()
            batch = data_cls.get_inputs(batch)
            batch = {key: value.to(model.device) for key, value in batch.items()}
            output = model(**batch)

            if teacher_batch:
                teacher_batch = data_cls.get_inputs(teacher_batch, teacher=True)
                teacher_batch = {key: value.to(model.device) for key, value in teacher_batch.items()}

            loss = loss_cls.get_loss(output, batch if not teacher_batch else teacher_batch)

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
                                                              label_map=args.label_map, tokenizer=student_tokenizer)

                if update_steps % args.save_steps:
                    torch.save(model.state_dict(), f'{saving_name}_last.pt')


def eval(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device

    tokenizer, model, _, _ = init_model(args)
    model.eval()

    writer = None
    _, val_dataloader = get_bc2gm_train_data(args, tokenizer, args.label2id, return_train=False, return_val=True)
    # eval
    eval_func = evaluate_ner_metrics if args.task_name == 'ner' else None
    evaluator = Evaluator(model, eval_func, val_dataloader, writer)
    _, metrics = evaluator.evaluate_and_write(0,
                                              label_map=args.label_map, tokenizer=tokenizer)

    print(metrics)


def measure_time(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device
    args.batch_size = 1

    tokenizer, model, teacher_tokenizer, teacher_model = init_model(args)
    model.eval()
    if teacher_model is not None:
        teacher_model.eval()

    _, val_dataloader = get_bc2gm_train_data(args, tokenizer, args.label2id, return_train=False, return_val=True)
    if teacher_tokenizer is not None:
        _, teacher_val_dataloader = get_bc2gm_train_data(args, teacher_tokenizer, args.label2id,
                                                         return_train=False, return_val=True)
    else:
        teacher_val_dataloader = val_dataloader

    tags_vocab = {value: key for key, value in args.label_map.items()}
    data_cls = NerData(tokenizer, teacher_tokenizer, tags_vocab) if args.task_name == 'ner' else Data()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # gpu warm-up
    for batch in tqdm(val_dataloader, desc="GPU warm-up"):
        batch = data_cls.get_inputs(batch)
        _ = model(**{key: value.to(device) for key, value in batch.items()})

    student_times = torch.zeros(len(val_dataloader))
    teacher_times = torch.zeros(len(val_dataloader))
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader, desc="Evaluation")):
            batch = data_cls.get_inputs(batch)
            batch = {key: value.to(device) for key, value in batch.items()}
            starter.record()
            _ = model(**batch)
            ender.record()
            torch.cuda.synchronize()
            student_times[i] = starter.elapsed_time(ender)

    if teacher_model is not None:
        with torch.no_grad():
            for i, batch in enumerate(tqdm(teacher_val_dataloader, desc="Teacher")):
                batch = data_cls.get_inputs(batch)
                batch = {key: value.to(device) for key, value in batch.items()}
                starter.record()
                _ = teacher_model(**batch)
                ender.record()
                torch.cuda.synchronize()
                teacher_times[i] = starter.elapsed_time(ender)

    print(f"Teacher time: mean = {teacher_times.mean()}, std = {teacher_times.std()}")
    print(f"Student time: mean = {student_times.mean()}, std = {student_times.std()}")


if __name__ == "__main__":
    args = setup_argparser().parse_args()
    if args.measure_time:
        measure_time(args)
    elif args.do_train:
        train(args)
    else:
        eval(args)

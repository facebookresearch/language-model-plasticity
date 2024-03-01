# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import math
import torch.nn.functional as F
import torch
from fairseq import utils
from fairseq import metrics
from . import FairseqCriterion, register_criterion

from functools import reduce
import logging
logger = logging.getLogger(__name__)

@register_criterion('squad')
class SquadCLSCriterion(FairseqCriterion):
    def __init__(self, task, classification_head, cls_alpha):
        super().__init__(task)
        self.cls_alpha = cls_alpha
        self.classification_head = classification_head

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--cls-alpha', default=1.0, type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    @classmethod
    def build_criterion(cls, args, task):
        return cls(task, task.squad_head_name, args.cls_alpha)

    def forward(self, model, sample, reduce=True):
        targets = sample['target']
        paragraph_mask = sample['paragraph_mask']

        (start_out, end_out, paragraph_mask), _ = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name=self.classification_head,
            paragraph_mask=paragraph_mask,
        )
        outs = (start_out, end_out)

        questions_mask = paragraph_mask.ne(1)
        paragraph_outs = [
            o.view(-1, o.size(1)).masked_fill(questions_mask, 0) for o in outs]
        outs = paragraph_outs

        outs = [F.log_softmax(o, dim=1).view(-1, o.size(1)) for o in outs]
        ignored_index = outs[0].size(1)

        with torch.no_grad():
            targets = [t.clamp(0, ignored_index) for t in targets]
            for idx in range(len(targets)):
                targets[idx][targets[idx] == self.padding_idx] = ignored_index

        sample_size = sample['nsentences'] * len(outs)
        loss = 0.0
        for t, o in zip(targets, outs):
            loss += F.nll_loss(o, t.view(-1), size_average=False,
                               ignore_index=ignored_index, reduce=reduce)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
            'extra_metrics': {'qna_start': [], 'qna_end': []}
        }
        with torch.no_grad():
            for g, o, t in zip(['qna_start', 'qna_end'], outs, sample['target']):
                t = t.squeeze(-1)
                pred_t = torch.argmax(o, dim=-1)
                tp = t.eq(pred_t).long().sum().item()
                tn = 0
                fp = t.size(0) - tp
                fn = 0
                logging_output[f'{g}_scores'] = (tp, tn, fp, fn)

        # logger.warning("loss: {0}, rank: {1}".format(logging_output['loss'], torch.distributed.get_rank()))
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        def calculate_acc_f1(scores):
            tp, tn, fp, fn = scores
            precision = tp / ((tp + fp) or 1.0)
            recall = tp / ((tp + fn) or 1.0)
            acc = (tp + tn) / ((tp + tn + fp + fn) or 1.0)
            f1 = 2 * precision * recall / ((precision + recall) or 1.0)
            return acc, f1
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        qna_start_scores = tuple(reduce(lambda q, w: (sum(x) for x in zip(q, w)), [
            log['qna_start_scores'] for log in logging_outputs]))
        qna_end_scores = tuple(reduce(lambda q, w: (sum(x) for x in zip(q, w)), [
            log['qna_end_scores'] for log in logging_outputs]))
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "qna_extractive_loss",
            loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        qna_start_acc, qna_start_f1 = calculate_acc_f1(qna_start_scores)
        qna_end_acc, qna_end_f1 = calculate_acc_f1(qna_end_scores)

        metrics.log_scalar(
            "qna_extractive_start_acc",
            100.0 * qna_start_acc,
            nsentences, round=3
        )
        metrics.log_scalar(
            "qna_extractive_end_acc",
            100.0 * qna_end_acc,
            nsentences, round=3
        )
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import numpy as np
import torch

from . import data_utils, FairseqDataset
from typing import List


class SquadDataset(FairseqDataset):
    """
    A wrapper around torch.utils.data.Dataset for monolingual data.
    Args:
        dataset (torch.utils.data.Dataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
    """

    def __init__(self, dataset1, dataset2,
                 ids, actual_txt,
                 idx_map, sizes1, sizes2,
                 dictionary, stride,
                 max_length, max_query_length,
                 labels=None, add_prev_output=False):
        self.dataset1, self.dataset2 = dataset1, dataset2
        self.sizes1, self.sizes2 = (sizes1), (sizes2)
        self.labels = (labels) if labels is not None else None
        self.ids = ids
        self.actual_txt = actual_txt
        self.idx_map = idx_map
        self.eos = dictionary.eos()
        self.bos = dictionary.bos()
        self.pad = dictionary.pad()
        self.shuffle = False
        self.stride = stride
        self.max_length = max_length
        self.max_query_length = max_query_length
        self.add_prev_output = add_prev_output

    def __getitem__(self, index):
        paragraph = self.dataset1[index][:-1]
        question = self.dataset2[index][:-1]
        lbl = self.labels[index] if self.labels is not None else None
        actual_txt = self.actual_txt[index]
        idx_map = [int(ii) for ii in self.idx_map[index]]
        if question.size(0) > self.max_query_length:
            question = question[:self.max_query_length]
        question_len = question.size(0) + 2  # account for cls, sep
        start_offset = 0
        doc_spans_text = []
        doc_spans = []
        max_tokens_for_doc = self.max_length - len(question) - 3
        assert max_tokens_for_doc > 0, f"max_tokens_for_doc is less than 0, for max length ({self.max_length}) and question length ({len(question)})"

        while start_offset < len(paragraph):
            length = len(paragraph) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans_text.append(
                paragraph[start_offset: start_offset+length].clone())
            doc_spans.append((start_offset, length))
            if start_offset + length == len(paragraph):
                break
            start_offset = start_offset + min(length, self.stride)

        if lbl is None or len(lbl) == 0:
            s, e = -1, -1
            result = 0
        else:
            s, e = lbl
            assert e >= s
            result = 1
        res = []
        for span_idx, span in enumerate(doc_spans_text):
            span_idx_map = []
            doc_start = doc_spans[span_idx][0]
            doc_end = doc_spans[span_idx][0] + doc_spans[span_idx][1] - 1
            span_idx_map = idx_map[doc_start: doc_end+1]
            out_of_span = False
            if not (s >= doc_start and e <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
            else:
                start_position = s - doc_start + question_len
                end_position = e - doc_start + question_len
            start_position = torch.tensor([start_position], dtype=torch.long)
            end_position = torch.tensor([end_position], dtype=torch.long)
            text, seg = self._join_sents(question, span)
            paragraph_mask = torch.zeros(text.shape).byte()
            paragraph_mask[question_len: -1] = 1
            target = (start_position, end_position)
            token_is_max_context = [0] * question_len
            for j in range(doc_spans[span_idx][1]):
                split_token_index = doc_spans[span_idx][0] + j
                is_max_context = self._check_is_max_context(
                    doc_spans, span_idx, split_token_index)
                token_is_max_context.append(is_max_context)
            res.append({'id': index, 'text': text, 'text_len': torch.numel(text), 'segment': seg, 'target': target, 'paragraph_mask': paragraph_mask,
                        'squad_ids': self.ids[index], 'actual_txt': np.asarray(actual_txt),
                        'idx_map': torch.tensor(span_idx_map, dtype=torch.long), 'HasAns': torch.tensor([result], dtype=torch.long),
                        'token_is_max_context': torch.tensor(token_is_max_context, dtype=torch.long)})
        return res

    def _join_sents(self, sent1, sent2):
        cls = sent1.new_full((1,), self.bos)
        sep = sent1.new_full((1,), self.eos)
        sent1 = torch.cat([cls, sent1, sep])
        sent2 = torch.cat([sent2, sep])
        text = torch.cat([sent1, sent2])
        segment1 = torch.zeros(sent1.size(0))
        segment2 = torch.ones(sent2.size(0))
        segment = torch.cat([segment1, segment2])

        return text, segment

    def __len__(self):
        return len(self.dataset1)

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        if not isinstance(samples[0], dict):
            samples = [s for sample in samples for s in sample]
        target_len = len(samples[0]['target'])
        target = [torch.stack([s['target'][i] for s in samples], dim=0)
                  for i in range(target_len)]
        HasAns = torch.stack([s['HasAns'] for s in samples], dim=0)
        actual_txt = [s['actual_txt'] for s in samples]
        idx_map = [s['idx_map'] for s in samples]
        token_is_max_context = [s['token_is_max_context'] for s in samples]
        paragraph_mask = data_utils.collate_tokens(
            [s['paragraph_mask'] for s in samples], self.pad,  left_pad=False)

        squad_dict = {
            'id': torch.tensor([s['id'] for s in samples], dtype=torch.long),
            'ntokens': sum(len(s['text']) for s in samples),
            'net_input': {
                'src_tokens': data_utils.collate_tokens(
                    [s['text'] for s in samples], self.pad, left_pad=False,
                ).long(),
                'src_lengths':
                    torch.tensor([s['text_len'] for s in samples]).long()
            },
            'target': target,
            'HasAns': HasAns,
            'nsentences': len(samples),
            'actual_txt': np.asarray(actual_txt),
            'idx_map': idx_map,
            'paragraph_mask': paragraph_mask,
            'possible_sentences': sum(int(s['target'][0] == 0) for s in samples),
            'squad_ids': [s['squad_ids'] for s in samples],
            'token_is_max_context': token_is_max_context
        }
        if self.add_prev_output:
            squad_dict["net_input"].update(
                prev_output_tokens=data_utils.collate_tokens(
                    [torch.roll(s['text'], 1) for s in samples], self.pad, left_pad=False,
                ).long())
        return squad_dict

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes1[index] + self.sizes2[index] + 1

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes1[index] + self.sizes2[index] + 1

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            return np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        indices = indices[np.argsort(self.sizes1[indices], kind='mergesort')]
        return indices[np.argsort(self.sizes2[indices], kind='mergesort')]

    def prefetch(self, indices):
        self.dataset1.prefetch(indices)
        self.dataset2.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
            hasattr(self.dataset1, 'supports_prefetch')
            and self.dataset1.supports_prefetch
            and hasattr(self.dataset2, 'supports_prefetch')
            and self.dataset2.supports_prefetch
        )

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span[0] + doc_span[1] - 1
            if position < doc_span[0]:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span[0]
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + \
                0.01 * doc_span[1]
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index


class SquadLeanDataset(SquadDataset):
    def __init__(self, dataset1, dataset2,
                 sizes1, sizes2,
                 dictionary, stride,
                 max_length, max_query_length,
                 labels=None, add_prev_output=False):
        self.dataset1, self.dataset2 = dataset1, dataset2
        self.sizes1, self.sizes2 = (sizes1), (sizes2)
        self.labels = labels
        self.eos = dictionary.eos()
        self.bos = dictionary.bos()
        self.pad = dictionary.pad()
        self.stride = stride
        self.max_length = max_length
        self.max_query_length = max_query_length
        self.add_prev_output = add_prev_output
        self.shuffle = False

    def __getitem__(self, index):
        paragraph = self.dataset1[index][:-1]
        question = self.dataset2[index][:-1]
        lbl = self.labels[index]
        if question.size(0) > self.max_query_length:
            question = question[:self.max_query_length]
        question_len = question.size(0) + 2  # account for cls, sep
        start_offset = 0
        doc_spans_text = []
        doc_spans = []
        max_tokens_for_doc = self.max_length - len(question) - 3
        assert max_tokens_for_doc > 0, f"max_tokens_for_doc is less than 0, for max length ({self.max_length}) and question length ({len(question)})"
        while start_offset < len(paragraph):
            length = len(paragraph) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans_text.append(
                paragraph[start_offset: start_offset+length].clone())
            doc_spans.append((start_offset, length))
            if start_offset + length == len(paragraph):
                break
            start_offset = start_offset + min(length, self.stride)

        if lbl is None or len(lbl) == 0:
            s, e = -1, -1
            result = 0
        else:
            s, e = lbl
            assert e >= s
            result = 1
        res = []
        for span_idx, span in enumerate(doc_spans_text):
            doc_start = doc_spans[span_idx][0]
            doc_end = doc_spans[span_idx][0] + doc_spans[span_idx][1] - 1
            out_of_span = False
            if not (s >= doc_start and e <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
            else:
                start_position = s - doc_start + question_len
                end_position = e - doc_start + question_len
            start_position = torch.tensor([start_position], dtype=torch.long)
            end_position = torch.tensor([end_position], dtype=torch.long)
            text, _ = self._join_sents(question, span)
            paragraph_mask = torch.zeros(text.shape).byte()
            paragraph_mask[question_len: -1] = 1
            target = (start_position, end_position)
            res.append({'id': index, 'text': text, 'text_len': torch.numel(text), 'target': target,
                        'paragraph_mask': paragraph_mask, 'HasAns': torch.tensor([result], dtype=torch.long), })
        return res

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        if not isinstance(samples[0], dict):
            samples = [s for sample in samples for s in sample]
        target_len = len(samples[0]['target'])
        target = [torch.stack([s['target'][i] for s in samples], dim=0)
                  for i in range(target_len)]
        HasAns = torch.stack([s['HasAns'] for s in samples], dim=0)
        paragraph_mask = data_utils.collate_tokens(
            [s['paragraph_mask'] for s in samples], self.pad,  left_pad=False)

        squad_dict = {
            'id': torch.tensor([s['id'] for s in samples], dtype=torch.long),
            'ntokens': sum(len(s['text']) for s in samples),
            'net_input': {
                'src_tokens': data_utils.collate_tokens(
                    [s['text'] for s in samples], self.pad, left_pad=False,
                ).long(),
                'src_lengths':
                    torch.tensor([s['text_len'] for s in samples]).long()
            },
            'target': target,
            'HasAns': HasAns,
            'nsentences': len(samples),
            'paragraph_mask': paragraph_mask,
        }
        if self.add_prev_output:
            squad_dict["net_input"].update(
                prev_output_tokens=data_utils.collate_tokens(
                    [torch.roll(s['text'], 1) for s in samples], self.pad, left_pad=False,
                ).long())
        return squad_dict
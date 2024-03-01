# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

from torch import nn


class SquadClassificationHead(nn.Module):
    """Head for Extractive MRC tasks."""

    def __init__(self, model_dim, predict_has_ans):
        super().__init__()
        self.qa_outputs = nn.Linear(model_dim, 2)
        self.reset_parameters()

    def reset_parameters(self):
        self.qa_outputs.weight.data.normal_(mean=0.0, std=0.02)
        self.qa_outputs.bias.data.zero_()

    def forward(self, features, paragraph_mask):
        logits = self.qa_outputs(features)
        if paragraph_mask.size(1) > features.size(1):
            paragraph_mask = paragraph_mask[:, :features.size(1)]
        assert [paragraph_mask[i].any() for i in range(paragraph_mask.size(0))]
        start, end = logits.split(1, dim=-1)
        return start, end, paragraph_mask
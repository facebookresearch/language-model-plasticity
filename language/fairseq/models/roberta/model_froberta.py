# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

"""
Frozen Roberta
"""
import ast
import logging

from fairseq.utils import safe_getattr
from fairseq.models import register_model, register_model_architecture


from .model import RobertaModel, base_architecture
from .model_iroberta import IRobertaModel, iroberta_base_architecture
from .model_xlmr import XLMRModel


logger = logging.getLogger(__name__)


@register_model("froberta")
class FRobertaModel(RobertaModel):
    def __init__(self, args, encoder):
        super(FRobertaModel, self).__init__(args, encoder)
        self._num_updates = 0
        if self.args.freeze_body:
            for param in self.parameters():
                param.requires_grad = False
            if not self.args.freeze_token_emb:
                self.encoder.sentence_encoder.embed_tokens.weight.requires_grad = True
            if not self.args.freeze_lm_head:
                self.encoder.lm_head.weight.requires_grad = True

        if self.args.freeze_token_emb:
            self.encoder.sentence_encoder.embed_tokens.weight.requires_grad = False
        
        if self.args.freeze_lm_head:
            self.encoder.lm_head.weight.requires_grad = False            
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(FRobertaModel, FRobertaModel).add_args(parser)
        parser.add_argument(
            "--freeze_token_emb", type=ast.literal_eval, default=False,
            help='Freeze the token embedding layer'
        )
        parser.add_argument(
            "--freeze_lm_head", type=ast.literal_eval, default=False,
            help='Freeze the lm_head, which is shared with token embedding layer'
        )
        parser.add_argument(
            "--freeze_body", type=ast.literal_eval, default=True,
            help='Freeze the transformer body, everything except the token embedding layer and lm_head'
            )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs,
    ): # resetting here will cause different runs to reset differently
        return super(FRobertaModel, self).forward(
            src_tokens,
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
            classification_head_name=classification_head_name,
            **kwargs,
        )


@register_model("firoberta")
class FIRobertaModel(IRobertaModel):
    def __init__(self, args, encoder):
        super(FIRobertaModel, self).__init__(args, encoder)
        if self.args.freeze_body:
            for param in self.parameters():
                param.requires_grad = False
            if not self.args.freeze_token_emb:
                self.encoder.sentence_encoder.embed_tokens.weight.requires_grad = True
            if not self.args.freeze_lm_head:
                self.encoder.lm_head.weight.requires_grad = True

        if self.args.freeze_token_emb:
            self.encoder.sentence_encoder.embed_tokens.weight.requires_grad = False
        
        if self.args.freeze_lm_head:
            self.encoder.lm_head.weight.requires_grad = False

        print('Require_grads for token embedding {}'.format(self.encoder.sentence_encoder.embed_tokens.weight.requires_grad))
        print('Require_grads for lm_head {}'.format(self.encoder.lm_head.weight.requires_grad))
        print('Require_grads for other layers {}'.format(self.encoder.sentence_encoder.layers[0].self_attn.k_proj.weight.requires_grad))

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(FIRobertaModel, FIRobertaModel).add_args(parser)
        parser.add_argument(
            "--freeze_token_emb", type=ast.literal_eval, default=False,
            help='Freeze the token embedding layer'
        )
        parser.add_argument(
            "--freeze_lm_head", type=ast.literal_eval, default=False,
            help='Freeze the lm_head, which is shared with token embedding layer'
        )
        parser.add_argument(
            "--freeze_body", type=ast.literal_eval, default=True,
            help='Freeze the transformer body, everything except the token embedding layer'
            )

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs,
    ):
        return super(FIRobertaModel, self).forward(
            src_tokens,
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
            classification_head_name=classification_head_name,
            **kwargs,
        )


@register_model("fxlmr")
class FXLMRModel(XLMRModel):
    def __init__(self, args, encoder):
        super(FXLMRModel, self).__init__(args, encoder)
        self._num_updates = 0
        if self.args.freeze_body:
            for param in self.parameters():
                param.requires_grad = False
            if not self.args.freeze_token_emb:
                self.encoder.sentence_encoder.embed_tokens.weight.requires_grad = True
            if not self.args.freeze_lm_head:
                self.encoder.lm_head.weight.requires_grad = True

        if self.args.freeze_token_emb:
            self.encoder.sentence_encoder.embed_tokens.weight.requires_grad = False
        
        if self.args.freeze_lm_head:
            self.encoder.lm_head.weight.requires_grad = False            
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(FRobertaModel, FRobertaModel).add_args(parser)
        parser.add_argument(
            "--freeze_token_emb", type=ast.literal_eval, default=False,
            help='Freeze the token embedding layer'
        )
        parser.add_argument(
            "--freeze_lm_head", type=ast.literal_eval, default=False,
            help='Freeze the lm_head, which is shared with token embedding layer'
        )
        parser.add_argument(
            "--freeze_body", type=ast.literal_eval, default=True,
            help='Freeze the transformer body, everything except the token embedding layer and lm_head'
            )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs,
    ): # resetting here will cause different runs to reset differently
        return super(FXLMRModel, self).forward(
            src_tokens,
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
            classification_head_name=classification_head_name,
            **kwargs,
        )


@register_model_architecture("froberta", "froberta_base")
def froberta_base_architecture(args):
    args.freeze_token_emb = safe_getattr(args, "freeze_token_emb", False)
    args.freeze_lm_head = safe_getattr(args, "freeze_lm_head", False)
    args.freeze_body = safe_getattr(args, "freeze_body", True)
    logger.warning("Freeze token embedding layer {}".format(args.freeze_token_emb))
    logger.warning("Freeze lm_head layer {}".format(args.freeze_lm_head))
    logger.warning("Freeze body {}".format(args.freeze_body))
    base_architecture(args)


@register_model_architecture("firoberta", "firoberta_base")
def firoberta_base_architecture(args):
    args.freeze_token_emb = safe_getattr(args, "freeze_token_emb", False)
    args.freeze_lm_head = safe_getattr(args, "freeze_lm_head", False)
    args.freeze_body = safe_getattr(args, "freeze_body", True)
    logger.warning("Freeze token embedding layer {}".format(args.freeze_token_emb))
    logger.warning("Freeze lm_head layer {}".format(args.freeze_lm_head))
    logger.warning("Freeze body {}".format(args.freeze_body))
    iroberta_base_architecture(args)


@register_model_architecture("fxlmr", "fxlmr_base")
def fxlmr_base_architecture(args):
    args.freeze_token_emb = safe_getattr(args, "freeze_token_emb", False)
    args.freeze_lm_head = safe_getattr(args, "freeze_lm_head", False)
    args.freeze_body = safe_getattr(args, "freeze_body", True)
    logger.warning("Freeze token embedding layer {}".format(args.freeze_token_emb))
    logger.warning("Freeze lm_head layer {}".format(args.freeze_lm_head))
    logger.warning("Freeze body {}".format(args.freeze_body))
    base_architecture(args)
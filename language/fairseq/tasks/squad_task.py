# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

from functools import reduce
import itertools
import numpy as np
import os
import torch
from scripts.eval_squad import eval_dataset
from torch.utils.data import ConcatDataset

from fairseq.data import (
    Dictionary,  # IndexedInMemoryDataset,
    SquadDataset,
    TokenBlockDataset,
    indexed_dataset,
    encoders,
    SortDataset,
    data_utils,
)

from fairseq.logging import  progress_bar
import numpy as np



from . import FairseqTask, register_task


@register_task('squad')
class SquadTask(FairseqTask):
    """
    Classify a sentence
    Args:
        dictionary (Dictionary): the dictionary for the input of the classifier
    The sentence classification task provides the following additional command-line
    arguments:
    .. argparse::
        :ref: fairseq.tasks.sentence_classification_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--data-files', type=str,
                            help='colon separated json data files to score (assumed to be in data dir)')
        parser.add_argument('--n-best-size', type=int, default=20)
        parser.add_argument('--max-length', type=int, default=384)
        parser.add_argument('--stride', type=int, default=128)
        parser.add_argument('--max-query-length', type=int, default=64)
        # parser.add_argument('--max-positions', type=int, default=512)
        # parser.add_argument('--version2', action='store_true', default=False)
        parser.add_argument('--model-dim', type=int, default=1024)
        parser.add_argument('--add-prev-output-tokens', action='store_true', default=False,
                            help='add prev_output_tokens to sample, used for encoder-decoder arch')
        parser.add_argument('--squad-validation-updates', type=int, default=-1)

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.args = args
        self.dictionary = dictionary
        self.bpe = encoders.build_bpe(args)
        self.mask = dictionary.index("<mask>")

    def train_updates_done(
            self,
            num_updates,
            model,
            ddp_world_size=0,
            ddp_rank=0,
            ddp_process_group=None,
            model_parallel_rank=0,
    ):
        """Hook function called after evey update."""
        assert len(self.args.valid_subset.split(",")) == len(self.args.data_files.split(':'))
        if self.args.squad_validation_updates > 0 and num_updates > 0 and num_updates % self.args.squad_validation_updates == 0:
            # print(self.args.valid_subset)
            for data_file, split_name in zip(
                    self.args.data_files.split(':'),
                    self.args.valid_subset.split(","),
            ):
                eval_dataset(
                    self,
                    model,
                    split_name,
                    data_file,
                    self.args,
                    # ddp_world_size,
                    # ddp_rank,
                    # ddp_process_group,
                    # model_parallel_rank,
                    prefix="updates_{0}".format(num_updates),
                    # language=split_name.split('_')[-1],
                )
        pass

    def begin_valid_epoch(
            self,
            epoch,
            model,
            ddp_world_size=0,
            ddp_rank=0,
            ddp_process_group=None,
            model_parallel_rank=0,
            step=None
    ):
        assert len(self.args.valid_subset.split(",")) == len(self.args.data_files.split(':'))

        # if epoch > 1:
        if epoch > -1:
            # print(self.args.valid_subset)
            for data_file, split_name in zip(
                    self.args.data_files.split(':'),
                    self.args.valid_subset.split(","),
            ):
                progress = progress_bar.progress_bar(
                    [0],
                    log_format=self.args.log_format,
                    log_interval=self.args.log_interval,
                    epoch=epoch,
                    prefix=f"valid on '{split_name}' subset",
                    default_log_format=("tqdm" if not self.args.no_progress_bar else "simple"),
                    wandb_project=(
                        self.args.wandb_project
                    ),
                    wandb_run_name=os.environ.get(
                        "WANDB_NAME", os.path.basename(self.args.save_dir)
                    ),
                )
                res = eval_dataset(
                    self,
                    model,
                    split_name,
                    data_file,
                    self.args,
                    # ddp_world_size,
                    # ddp_rank,
                    # ddp_process_group,
                    # model_parallel_rank,
                    prefix="epoch_{0}".format(epoch),
                    # language=split_name.split('_')[-1],
                )
                import json
                [_ for _ in progress]
                print('*' * 80)
                print(epoch, step)
                progress.log(json.loads(str(res)[2:-3]), tag=split_name, step=epoch)
                # return res
        pass

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return self.dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
        # add <MASK> token
        dictionary.add_symbol('<mask>')
        print('| dictionary: {} types'.format(len(dictionary)))

        return cls(args, dictionary)

    def build_model(self, args):
        model = super().build_model(args)
        model.register_squad_head(
            name="squad_span_classification_head",
            predict_has_ans=False,
        )
        return model

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        loaded_datasets = [[], []]
        loaded_labels = []
        loaded_ids = []
        loaded_raw_actual_text = []
        loaded_tok_to_orig_index = []
        stop = False
        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            base_path = os.path.join(self.args.data, split_k)
            path1 = os.path.join(base_path + '_1')
            path2 = os.path.join(base_path + '_2')

            for path, datasets in zip([path1, path2], loaded_datasets):
                # if IndexedInMemoryDataset.exists(path):
                ds = indexed_dataset.make_dataset(path,
                                                  impl='mmap',
                                                  fix_lua_indexing=True,
                                                  dictionary=self.dictionary,
                                                  )
                if ds is None:
                    if k > 0:
                        stop = True
                        break
                    else:
                        raise FileNotFoundError(
                            'Dataset not found: {} ({}) {}'.format(split, self.args.data, path))

                datasets.append(
                    TokenBlockDataset(
                        ds,
                        ds.sizes,
                        512 - 1,  # one less for <s>
                        pad=self.dictionary.pad(),
                        eos=self.dictionary.eos(),
                        break_mode='eos',
                    ))

            if stop:
                break
            if split != 'test':
                with open(base_path + '.lbl', 'r') as lbl_f:
                    lines = lbl_f.readlines()
                    for line in lines:
                        lbls = [int(x) for x in line.strip().split()]
                        answers = [lbls[0], lbls[1]]

                        loaded_labels.append(answers)
            else:
                loaded_labels = None

            with open(base_path + '_3.txt', 'r') as act_f:
                lines = act_f.readlines()
                for line in lines:
                    loaded_raw_actual_text.append(line.strip())
            with open(base_path + '_4.txt', 'r') as idx_map_f:
                lines = idx_map_f.readlines()
                for line in lines:
                    idx_map = [int(x) for x in line.strip().split()]
                    loaded_tok_to_orig_index.append(idx_map)

            with open(base_path + '.id', 'r') as id_f:
                loaded_ids.extend([id.strip() for id in id_f.readlines()])

            print('| {} {} {} examples'.format(
                self.args.data, split_k, len(loaded_datasets[0][-1])))

            if not combine:
                break

        if len(loaded_datasets[0]) == 1:
            dataset1 = loaded_datasets[0][0]
            dataset2 = loaded_datasets[1][0]
            sizes1 = dataset1.sizes
            sizes2 = dataset2.sizes
        else:
            dataset1 = ConcatDataset(loaded_datasets[0])
            dataset2 = ConcatDataset(loaded_datasets[1])
            sizes1 = np.concatenate([ds.sizes for ds in loaded_datasets[0]])
            sizes2 = np.concatenate([ds.sizes for ds in loaded_datasets[1]])

        if split != 'test':
            assert len(dataset1) == len(loaded_labels)
            assert len(dataset2) == len(loaded_labels)

        dataset = SquadDataset(
            dataset1, dataset2,
            loaded_ids, loaded_raw_actual_text,
            loaded_tok_to_orig_index, sizes1,
            sizes2, self.dictionary, self.args.stride,
            self.args.max_length, self.args.max_query_length,
            labels=loaded_labels, add_prev_output=self.args.add_prev_output_tokens
        )

        with data_utils.numpy_seed(self.args.seed):
            dataset = SortDataset(
                dataset,
                # shuffle
                sort_order=[np.random.permutation(len(dataset))],
            )

        self.datasets[split] = dataset

    def max_positions(self):
        return self.args.max_positions

    def get_loss(self, model, criterion, sample, is_valid=False):
        loss, sample_size, logging_output, outs = criterion(model, sample)

        return loss, sample_size, logging_output

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    @property
    def squad_head_name(self):
        return "squad_span_classification_head"
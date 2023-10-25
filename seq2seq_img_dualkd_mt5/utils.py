import itertools
import json
import linecache
import math
import os
import h5py
import pickle
import socket
import glob
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from torch import nn
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader

from sentence_splitter import add_newline_to_end_of_each_sentence
from transformers import BartTokenizer, EvalPrediction, PreTrainedTokenizer, T5Tokenizer, BertTokenizer, RobertaTokenizer
from transformers.file_utils import cached_property
from transformers.models.bart.modeling_bart import shift_tokens_right
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


try:
    from fairseq.data.data_utils import batch_by_size

    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def calculate_bleu(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {"bleu": round(corpus_bleu(output_lns, [refs_lns], **kwargs).score, 4)}


def build_compute_metrics_fn(task_name: str, tokenizer: PreTrainedTokenizer, data_args) -> Callable[[EvalPrediction], Dict]:
    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)

    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        predictions = pred.predictions
        label_ids = pred.label_ids
        predictions[..., 0] = tokenizer.pad_token_id
        predictions[predictions == -100] = tokenizer.pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = lmap(str.strip, pred_str)
        label_str = lmap(str.strip, label_str)
        return pred_str, label_str

    def summarization_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        rouge: Dict = calculate_rouge(
            pred_str, label_str,
            rouge_lang=data_args.rouge_lang,
        )
        summ_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        rouge.update({"gen_len": summ_len})
        return rouge

    def translation_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        bleu: Dict = calculate_bleu(pred_str, label_str)
        gen_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        bleu.update({"gen_len": gen_len})
        return bleu

    compute_metrics_fn = summarization_metrics if "summarization" in task_name else translation_metrics
    return compute_metrics_fn


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class MultiDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        **dataset_kwargs
    ):
        super().__init__()
        assert "upsampling_factor" in dataset_kwargs, "upsampling_factor required"
        assert "total_batch_size" in dataset_kwargs, "total_batch_size required"
        assert "actual_batch_size" in dataset_kwargs, "actual_batch_size required"
        assert "gradient_accum" in dataset_kwargs, "gradient_accum required"
        assert "is_distributed" in dataset_kwargs, "is_distributed required"
        assert "dataset_class" in dataset_kwargs, "dataset_class required"
        
        self.dataloaders = []
        self.total_batch_size = dataset_kwargs.pop("total_batch_size")
        dataset_class = dataset_kwargs.pop("dataset_class")
        extension = "tokenized" if dataset_class == TokenizedDataset else "source"
        # identify all source training files
        datasets = []
        delimiter = f'_{type_path}.{extension}'

        for src_file in glob.glob(os.path.join(data_dir, f'*{type_path}.{extension}')):
            id = os.path.basename(src_file).rsplit(delimiter, 1)[0]
            type_path = "".join(os.path.basename(src_file).rsplit(f".{extension}", 1))
            dataset = dataset_class(
                tokenizer,
                type_path=type_path,
                data_dir=data_dir,
                n_obs=n_obs,
                max_target_length=max_target_length,
                max_source_length=max_source_length,
                prefix=prefix,
                data_id=id
            )
            datasets.append(dataset)
            train_sampler = RandomSampler(dataset)
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                sampler=train_sampler,
                collate_fn=lambda batch: batch
            )
            self.dataloaders.append(dataloader)
        assert len(self.dataloaders) > 1, "multiple source/target filepairs required for MultiDataset"
        # compute effective length of this dataset and the sampling probabilities
        logger.info(f"Found datasets: {len(self.dataloaders)}")
        upsampling_factor = dataset_kwargs.get("upsampling_factor")
        datapoint_counts = np.array([len(dataset) for dataset in datasets])
        logger.info(f"Total datapoints: {np.sum(datapoint_counts)}")
        
        datapoint_probs = datapoint_counts / datapoint_counts.sum()
        smoothed_probs = datapoint_probs ** upsampling_factor

        self.sampling_probs = smoothed_probs / smoothed_probs.sum()
        self.effective_length = int(np.sum(datapoint_counts * self.sampling_probs))
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]

        is_distributed = dataset_kwargs.get("is_distributed")
        actual_batch_size = dataset_kwargs.get("actual_batch_size")
        gradient_accum = dataset_kwargs.get("gradient_accum")
        self.per_gpu_effective_batch_size = actual_batch_size * gradient_accum
            
        rank = int(os.environ.get("RANK")) if is_distributed else -1
        self.pos_shift_count = rank * self.per_gpu_effective_batch_size
        logger.info(f'Rank: {rank}, shifting required: {self.pos_shift_count}')
                
        self.current_dataset_idx = -1
        self.current_loader_count = 0

    
    def shift_iterator(self, idx, shift_count):
        if shift_count <= 0:
            return
        iterator = self.iterators[idx]
        for _ in range(shift_count):
            try:
                next(iterator)
            except StopIteration:
                dataloader = self.dataloaders[idx]
                iterator = iter(dataloader)
                
        self.iterators[idx] = iterator

    def __len__(self):
        return self.effective_length

    def __getitem__(self, index):
        if self.current_loader_count == 0:
            self.current_dataset_idx = np.random.choice(range(len(self.dataloaders)), p=self.sampling_probs)
            # start of a new effective batch, shift to appropriate pos
            self.shift_iterator(self.current_dataset_idx, self.pos_shift_count)
            
        iterator = self.iterators[self.current_dataset_idx]
        self.current_loader_count = (self.current_loader_count + 1) % self.total_batch_size
        
        try:
            datapoint = next(iterator)
        except StopIteration:
            dataloader = self.dataloaders[self.current_dataset_idx]
            self.iterators[self.current_dataset_idx] = iter(dataloader)
            datapoint = next(self.iterators[self.current_dataset_idx])

        if self.current_loader_count == self.per_gpu_effective_batch_size:
            # taken allocated datapoints from this effective batch, move to the start of next batch
            self.shift_iterator(self.current_dataset_idx, self.total_batch_size - self.current_loader_count - self.pos_shift_count)
            self.current_loader_count = 0

        return datapoint[0]

class UnistageCrosslingualDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        **dataset_kwargs
    ):
        super().__init__()
        assert "per_lang_batch_size" in dataset_kwargs, "per_lang_batch_size required"
        assert "upsampling_factor" in dataset_kwargs, "upsampling_factor required"
        assert "total_batch_size" in dataset_kwargs, "total_batch_size required"
        assert "actual_batch_size" in dataset_kwargs, "actual_batch_size required"
        assert "gradient_accum" in dataset_kwargs, "gradient_accum required"
        assert "is_distributed" in dataset_kwargs, "is_distributed required"
        assert "dataset_class" in dataset_kwargs, "dataset_class required"
        
        logger.info("Using cross lingual dataset with unistage sampling")
        self.dataloaders = []
        self.total_batch_size = dataset_kwargs.pop("total_batch_size")
        dataset_class = dataset_kwargs.pop("dataset_class")
        extension = "tokenized" if dataset_class == TokenizedDataset else "source"
        # identify all source training files
        datasets = []
        delimiter = f'_{type_path}.{extension}'

        for src_file in glob.glob(os.path.join(data_dir, f'*{type_path}.{extension}')):
            id = os.path.basename(src_file).rsplit(delimiter, 1)[0]
            tgt_lang = id.split("-")[1]
            type_path = "".join(os.path.basename(src_file).rsplit(f".{extension}", 1))
            dataset = dataset_class(
                tokenizer,
                type_path=type_path,
                data_dir=data_dir,
                n_obs=n_obs,
                max_target_length=max_target_length,
                max_source_length=max_source_length,
                prefix=prefix,
                data_id=tgt_lang
            )
            datasets.append(dataset)
            train_sampler = RandomSampler(dataset)
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                sampler=train_sampler,
                collate_fn=lambda batch: batch
            )
            self.dataloaders.append(dataloader)
            
        assert len(self.dataloaders) > 1, "multiple source/target filepairs required for MultiDataset"
        # compute effective length of this dataset and the sampling probabilities
        logger.info(f"Found datasets: {len(self.dataloaders)}")
        upsampling_factor = dataset_kwargs.get("upsampling_factor")
        datapoint_counts = np.array([len(dataset) for dataset in datasets])
        logger.info(f"Total datapoints: {np.sum(datapoint_counts)}")
        
        datapoint_probs = datapoint_counts / datapoint_counts.sum()
        smoothed_probs = datapoint_probs ** upsampling_factor

        self.sampling_probs = smoothed_probs / smoothed_probs.sum()
        self.effective_length = int(np.sum(datapoint_counts * self.sampling_probs))
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]

        is_distributed = dataset_kwargs.get("is_distributed")
        actual_batch_size = dataset_kwargs.get("actual_batch_size")
        gradient_accum = dataset_kwargs.get("gradient_accum")
        self.per_lang_batch_size = dataset_kwargs.get("per_lang_batch_size")
        
        self.per_gpu_effective_batch_size = actual_batch_size * gradient_accum
        self.per_gpu_lang_batch_size = self.per_lang_batch_size // (self.total_batch_size // self.per_gpu_effective_batch_size)

        assert self.total_batch_size % self.per_lang_batch_size == 0, "total_batch_size must be divisible by per_lang_batch_size"
        rank = int(os.environ.get("RANK")) if is_distributed else -1
        self.pos_shift_count = rank * self.per_gpu_lang_batch_size
        logger.info(f'Rank: {rank}, shifting required: {self.pos_shift_count}')
        logger.info(f"Effective length: {self.effective_length}")
        
                
        self.current_dataset_idx = -1
        self.current_loader_count = 0

    
    def shift_iterator(self, idx, shift_count):
        if shift_count <= 0:
            return
        iterator = self.iterators[idx]
        for _ in range(shift_count):
            try:
                next(iterator)
            except StopIteration:
                dataloader = self.dataloaders[idx]
                iterator = iter(dataloader)
                
        self.iterators[idx] = iterator

    def __len__(self):
        return self.effective_length

    def __getitem__(self, index):
        if self.current_loader_count == 0:
            self.current_dataset_idx = np.random.choice(range(len(self.dataloaders)), p=self.sampling_probs)
            # start of a new effective batch, shift to appropriate pos
            self.shift_iterator(self.current_dataset_idx, self.pos_shift_count)
            
        iterator = self.iterators[self.current_dataset_idx]
        self.current_loader_count = (self.current_loader_count + 1) % self.per_gpu_lang_batch_size
        
        try:
            datapoint = next(iterator)
        except StopIteration:
            dataloader = self.dataloaders[self.current_dataset_idx]
            self.iterators[self.current_dataset_idx] = iter(dataloader)
            datapoint = next(self.iterators[self.current_dataset_idx])

        if self.current_loader_count == 0:
            # taken allocated datapoints from this effective batch, move to the start of next batch
            self.shift_iterator(self.current_dataset_idx, self.per_lang_batch_size - self.per_gpu_lang_batch_size - self.pos_shift_count)

        return datapoint[0]


class CrosslingualDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        **dataset_kwargs
    ):
        super().__init__()
        assert "multistage_upsampling_factors" in dataset_kwargs, "multistage_upsampling_factors required"
        assert "per_lang_batch_size" in dataset_kwargs, "per_lang_batch_size required"
        assert "total_batch_size" in dataset_kwargs, "total_batch_size required"
        assert "actual_batch_size" in dataset_kwargs, "actual_batch_size required"
        assert "gradient_accum" in dataset_kwargs, "gradient_accum required"
        assert "is_distributed" in dataset_kwargs, "is_distributed required"
        assert "dataset_class" in dataset_kwargs, "dataset_class required"
        
        logger.info("Using cross lingual dataset")
        self.dataloaders = {}
        self.total_batch_size = dataset_kwargs.pop("total_batch_size")
        dataset_class = dataset_kwargs.pop("dataset_class")
        extension = "tokenized" if dataset_class == TokenizedDataset else "source"
        
        # identify all source training files
        forward_datapoint_counts = {}
        reverse_datapoint_counts = {}
        delimiter = f'_{type_path}.{extension}'

        # identify all languages first
        data_langs = set()
        for src_file in glob.glob(os.path.join(data_dir, f'*{type_path}.{extension}')):
            id = os.path.basename(src_file).rsplit(delimiter, 1)[0]
            data_langs.update(id.split("-"))
        
        total_dataset_count = 0
        # now create datasets according to language pairs
        for src_lang in data_langs:
            self.dataloaders[src_lang] = {}
            forward_datapoint_counts[src_lang] = {}

            for tgt_lang in data_langs:
                src_file = os.path.join(
                    data_dir,
                    f"{src_lang}-{tgt_lang}{delimiter}"
                )
                if not os.path.isfile(src_file):
                    continue
                
                type_path = "".join(os.path.basename(src_file).rsplit(f".{extension}", 1))
                dataset = dataset_class(
                    tokenizer,
                    type_path=type_path,
                    data_dir=data_dir,
                    n_obs=n_obs,
                    max_target_length=max_target_length,
                    max_source_length=max_source_length,
                    prefix=prefix,
                    data_id=tgt_lang
                )

                train_sampler = RandomSampler(dataset)
                dataloader = DataLoader(
                    dataset,
                    batch_size=1,
                    sampler=train_sampler,
                    collate_fn=lambda batch: batch
                )
                self.dataloaders[src_lang][tgt_lang] = dataloader
                forward_datapoint_counts[src_lang][tgt_lang] = len(dataset)
                
                source_data = reverse_datapoint_counts.get(tgt_lang, {})
                source_data[src_lang] = len(dataset)
                reverse_datapoint_counts[tgt_lang] = source_data

                total_dataset_count += 1

            if not self.dataloaders[src_lang]:
                self.dataloaders.pop(src_lang)
                forward_datapoint_counts.pop(src_lang)

        # compute effective length of this dataset and the sampling probabilities
        logger.info(f"Found datasets: {total_dataset_count}")
        multistage_upsampling_factors = dataset_kwargs.get("multistage_upsampling_factors")
        logger.info(f"Total datapoints: {np.sum(v for l in forward_datapoint_counts.values() for v in l.values())}")
        
        # avoiding matrix operations for easy traversing later
        # first stage datapoint probs

        def get_sampling_probs(data_matrix):
            first_stage_datapoint_counts = {
                k: np.sum(list(v.values())) for k, v in data_matrix.items()
            }
        
            first_stage_datapoint_probs = {
                k: l ** multistage_upsampling_factors[0] 
                    for k, l in zip(
                        first_stage_datapoint_counts.keys(),
                        list(first_stage_datapoint_counts.values()) / np.sum(list(first_stage_datapoint_counts.values()))
                    )
            }

            first_stage_sampling_probs = {
                k: l for k, l in zip(
                    first_stage_datapoint_probs.keys(),
                    list(first_stage_datapoint_probs.values()) / np.sum(list(first_stage_datapoint_probs.values()))  
                )
            }

    
            second_stage_sampling_probs = {}
            langwise_effective_lengths = []
            
            # second stage datapoint probs
            for src_lang, tgt_data_stats in data_matrix.items():
                lang_datapoint_probs = {
                    k: l ** multistage_upsampling_factors[1]
                        for k, l in zip(
                            tgt_data_stats.keys(),
                            list(tgt_data_stats.values()) / np.sum(list(tgt_data_stats.values()))
                        )
                }
                lang_sampling_probs = {
                    k: l for k, l in zip(
                        lang_datapoint_probs.keys(),
                        list(lang_datapoint_probs.values()) / np.sum(list(lang_datapoint_probs.values()))
                    )
                }
                second_stage_sampling_probs[src_lang] = lang_sampling_probs 
                langwise_effective_lengths.append(
                    np.sum(
                        np.array(list(tgt_data_stats.values())) * np.array(list(lang_sampling_probs.values()))
                    )
                )

            effective_length = int(
                np.sum(
                    langwise_effective_lengths * np.array(list(first_stage_sampling_probs.values()))
                )
            )

            return first_stage_sampling_probs, second_stage_sampling_probs, effective_length

        (
            self.src_first_stage_sampling_probs, 
            self.src_second_stage_sampling_probs, 
            src_effective_length 
        ) = get_sampling_probs(forward_datapoint_counts)

        (
            self.tgt_first_stage_sampling_probs, 
            self.tgt_second_stage_sampling_probs, 
            tgt_effective_length 
        ) = get_sampling_probs(reverse_datapoint_counts)

        self.effective_length = (src_effective_length + tgt_effective_length) // 2
        self.iterators = {}
        for src_lang, tgt_dataloaders in self.dataloaders.items():
            self.iterators[src_lang] = {k: iter(l) for k, l in tgt_dataloaders.items()}
            

        is_distributed = dataset_kwargs.get("is_distributed")
        actual_batch_size = dataset_kwargs.get("actual_batch_size")
        gradient_accum = dataset_kwargs.get("gradient_accum")
        self.per_lang_batch_size = dataset_kwargs.get("per_lang_batch_size")
        self.per_gpu_effective_batch_size = actual_batch_size * gradient_accum
        self.per_gpu_lang_batch_size = self.per_lang_batch_size // (self.total_batch_size // self.per_gpu_effective_batch_size)

        assert self.total_batch_size % self.per_lang_batch_size == 0, "total_batch_size must be divisible by per_lang_batch_size"
        rank = int(os.environ.get("RANK")) if is_distributed else -1
        self.pos_shift_count = rank * self.per_gpu_lang_batch_size
        logger.info(f'Rank: {rank}, shifting required: {self.pos_shift_count}')
        logger.info(f"Effective length: {self.effective_length}")

        # for choosing whether src or tgt will be fixed when taking a batch
        self.choice = -1
                
        self.current_src_lang = None
        self.current_tgt_lang = None
        self.current_src_loader_count = 0
        self.current_tgt_loader_count = 0


    def shift_iterator(self, src_lang, tgt_lang, shift_count):
        if shift_count <= 0:
            return

        iterator = self.iterators[src_lang][tgt_lang]
        for _ in range(shift_count):
            try:
                next(iterator)
            except StopIteration:
                dataloader = self.dataloaders[src_lang][tgt_lang]
                iterator = iter(dataloader)
                
        self.iterators[src_lang][tgt_lang] = iterator

    def __len__(self):
        return self.effective_length

    def __getitem__(self, index):
        if self.current_src_loader_count + self.current_tgt_loader_count == 0:
            self.choice = np.random.choice([0, 1], p=[0.5, 0.5])

        if self.choice == 0:
            # we keep src fixed
            if self.current_src_loader_count == 0:
                self.current_src_lang = np.random.choice(
                    list(self.src_first_stage_sampling_probs.keys()),
                    p=list(self.src_first_stage_sampling_probs.values())
                )
                
            if self.current_tgt_loader_count == 0:
                self.current_tgt_lang = np.random.choice(
                    list(self.src_second_stage_sampling_probs[self.current_src_lang].keys()),
                    p=list(self.src_second_stage_sampling_probs[self.current_src_lang].values())
                )
                # start of a new effective batch, shift to appropriate pos
                self.shift_iterator(
                    self.current_src_lang,
                    self.current_tgt_lang,
                    self.pos_shift_count
                )

            self.current_src_loader_count = (self.current_src_loader_count + 1) % self.total_batch_size
            self.current_tgt_loader_count = (self.current_tgt_loader_count + 1) % self.per_gpu_lang_batch_size
            

        elif self.choice == 1:
            # we keep tgt fixed
            if self.current_tgt_loader_count == 0:
                self.current_tgt_lang = np.random.choice(
                    list(self.tgt_first_stage_sampling_probs.keys()),
                    p=list(self.tgt_first_stage_sampling_probs.values())
                )
                
            if self.current_src_loader_count == 0:
                self.current_src_lang = np.random.choice(
                    list(self.tgt_second_stage_sampling_probs[self.current_tgt_lang].keys()),
                    p=list(self.tgt_second_stage_sampling_probs[self.current_tgt_lang].values())
                )
                # start of a new effective batch, shift to appropriate pos
                self.shift_iterator(
                    self.current_src_lang,
                    self.current_tgt_lang,
                    self.pos_shift_count
                )

            self.current_tgt_loader_count = (self.current_tgt_loader_count + 1) % self.total_batch_size
            self.current_src_loader_count = (self.current_src_loader_count + 1) % self.per_gpu_lang_batch_size


        iterator = self.iterators[self.current_src_lang][self.current_tgt_lang]            
        
        try:
            datapoint = next(iterator)
        except StopIteration:
            dataloader = self.dataloaders[self.current_src_lang][self.current_tgt_lang]
            self.iterators[self.current_src_lang][self.current_tgt_lang] = iter(dataloader)
            datapoint = next(self.iterators[self.current_src_lang][self.current_tgt_lang])

        if (
                (self.choice == 0 and self.current_tgt_loader_count == 0) or
                (self.choice == 1 and self.current_src_loader_count == 0)
        ):
            # taken allocated datapoints from this effective batch, move to the start of next batch
            self.shift_iterator(
                self.current_src_lang, 
                self.current_tgt_lang,
                self.per_lang_batch_size - self.per_gpu_lang_batch_size - self.pos_shift_count
            )
            

        return datapoint[0]




class TokenizedDataset(Dataset):
    """Dataset to load tokenized data. Backwards compatible with AbstractSeq2SeqDataset"""

    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        **dataset_kwargs
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".tokenized")
        self.length = self.get_lc(self.src_file)
        
        if n_obs is not None and n_obs != -1:
            self.length = min(n_obs, self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index + 1  # linecache starts at 1
        source_line = linecache.getline(str(self.src_file), index).rstrip("\n")
        return json.loads(source_line)
        
    @staticmethod
    def get_lc(data_file):
        with open(data_file) as f:
            for lc, _ in enumerate(f, 1):
                pass
        return lc


class TokenizedDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        processed_batch = {}
        
        for k in batch[0]:
            processed_batch[k] = torch.stack(
                [torch.tensor(x[k]).squeeze() for x in batch]
            )

        return processed_batch

        
class AbstractSeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        **dataset_kwargs
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.src_file2 = Path(data_dir).joinpath(type_path + ".source2")
        self.tgt_file2 = Path(data_dir).joinpath(type_path + ".target2")

        self.img_file = Path(data_dir).joinpath(type_path + ".tag")
        self.image_feature_path = Path(data_dir).joinpath(type_path + ".boxes36.h5")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")
        if os.path.exists(self.len_file):
            self.src_lens = pickle_load(self.len_file)
            self.used_char_len = False
        else:
            self.src_lens = self.get_char_lens_max(self.src_file, self.src_file2)
            self.used_char_len = True
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        if n_obs is not None and n_obs != -1:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs
        dataset_kwargs.update({"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {})

    def __len__(self):
        return len(self.src_lens)
    @staticmethod
    def get_char_lens_max(data_file, data_file2):
        return [max(len(x1), len(x2)) for x1, x2 in zip(Path(data_file).open().readlines(), Path(data_file2).open().readlines())]

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert not self.used_char_len, "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.src_lens[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [max(self.src_lens[i] for i in batch) * len(batch) for batch in shuffled_batches]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")


class LegacySeq2SeqDataset(AbstractSeq2SeqDataset):
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """Call tokenizer on src and tgt_lines"""
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        source_inputs = self.encode_line(self.tokenizer, source_line, self.max_source_length)
        target_inputs = self.encode_line(self.tokenizer, tgt_line, self.max_target_length)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": target_ids,
        }

    def encode_line(self, tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"):
        """Only used by LegacyDataset"""
        return tokenizer(
            [line],
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            **self.dataset_kwargs,
        )

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["labels"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": y,
        }
        return batch


class Seq2SeqDataset(AbstractSeq2SeqDataset):
    """A dataset that calls prepare_seq2seq_batch."""

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        data_id = self.dataset_kwargs.get("data_id", None)
        img_line = linecache.getline(str(self.img_file), index).rstrip("\n")
        img_id = img_line.strip().split('\t')[:3]

        source_line2 = self.prefix + linecache.getline(str(self.src_file2), index).rstrip("\n")
        tgt_line2 = linecache.getline(str(self.tgt_file2), index).rstrip("\n")

        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
#        return {"tgt_texts": tgt_line, "src_texts": source_line, "data_id": data_id}
        return {"tgt_texts": tgt_line, "src_texts": source_line, "data_id": data_id, "img_ids": img_id, "img_feature_path": self.image_feature_path, "tgt_texts2": tgt_line2, "src_texts2": source_line2}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""
        batch_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
            **self.dataset_kwargs,
        ).data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding


class Seq2SeqDataCollator:
    def __init__(self, tokenizer, data_args, padding=None, tpu_num_cores=None, max_img_len=108):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        self.tpu_num_cores = tpu_num_cores
        self.max_img_len = max_img_len
        self.dataset_kwargs = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
        if data_args.src_lang is not None:
            self.dataset_kwargs["src_lang"] = data_args.src_lang
        if data_args.tgt_lang is not None:
            self.dataset_kwargs["tgt_lang"] = data_args.tgt_lang
        self.is_bert_based = self.tokenizer.cls_token is not None
        self.padding = padding if padding is not None else ("max_length" if self.tpu_num_cores is not None else "longest")

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        init_token_id = self.pad_token_id
        #print("self.pad_token_id", self.pad_token_id)
        lang_idx = None
        if "data_id" in batch[0]:
            init_token = batch[0]["data_id"]
            # get id from vocab
            if hasattr(self.data_args, "langid_map"):
                mapped_data = self.data_args.langid_map.get(init_token, None)
                if mapped_data:
                    lang_idx, mapped_token = mapped_data
                    init_token_id = self.tokenizer._convert_token_to_id(mapped_token)
                else:
                    logger.error(f"Unknown langid: {init_token}")

        if self.is_bert_based:
            batch = self._bert_encode(batch)
        elif hasattr(self.tokenizer, "prepare_seq2seq_batch"):
            image_feature = []
            img = np.zeros([len(batch), self.max_img_len, 2048])
            img_len = []
            n_boxes = 36
            for k, item in enumerate(batch):
                f = item["img_feature_path"]
                #print("h5 file,", f)
                f = h5py.File(f, 'r')
                i = 0
                for j, img_id in enumerate(item["img_ids"]):
                    feats = np.zeros(shape=(n_boxes, 2048), dtype=np.float32)
                    if img_id in f.keys():
                        f[f'{img_id}/features'].read_direct(feats)
                        if i == 0:
                            image_feature = feats
                        else:
                            image_feature = np.concatenate((image_feature, feats), axis=0)
                        i += 1
                if i == 0:
                    image_feature = np.zeros(shape=(1, 2048), dtype=np.float32)
                    img_len.append(image_feature.shape[0])
                else:
                    img_len.append(image_feature.shape[0])
                img[k][:image_feature.shape[0]] = image_feature
#                img_len.append(image_feature.shape[0])
            img = img[:,:max(img_len)]
            img = torch.tensor(img, dtype=torch.float32)
#            batch = self._encode(batch)
            batch, batch2 = self._encode_two(batch)

            input_ids, attention_mask, labels, input_ids2, attention_mask2, labels2 = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
                batch2["input_ids"],
                batch2["attention_mask"],
                batch2["labels"],
            )
            # labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            input_ids = torch.stack([x["input_ids"] for x in batch])
            attention_mask = torch.stack([x["attention_mask"] for x in batch])
            labels = torch.stack([x["labels"] for x in batch])

            labels = trim_batch(labels, self.pad_token_id)
            input_ids, attention_mask = trim_batch(input_ids, self.pad_token_id, attention_mask=attention_mask)

        if isinstance(self.tokenizer, T5Tokenizer):
            decoder_input_ids = self._shift_right_t5(labels, init_token_id)
            decoder_input_ids2 = self._shift_right_t5(labels2, init_token_id)

            if self.data_args.use_langid_prefix:
                input_ids = self._shift_right_t5(input_ids, init_token_id)
                input_ids2 = self._shift_right_t5(input_ids2, init_token_id)
            # if self.data_args.use_langid:
            #     lang_ids = torch.empty(
            #         labels.shape,
            #         dtype=labels.dtype).fill_(lang_idx)
        elif self.is_bert_based:
            # bert based models will automatically add the [CLS] token
            pass
        else:
            decoder_input_ids = shift_tokens_right(labels, self.pad_token_id)

        
        if not self.is_bert_based:
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": decoder_input_ids,
                "labels": labels,
                "init_token_id": init_token_id,
                "image_features": img,
                "image_len": img_len,
                # "lang_ids": lang_ids
                "input_ids2": input_ids2,
                "attention_mask2": attention_mask2,
                "decoder_input_ids2": decoder_input_ids2,
                "labels2": labels2,
            }
            
        return batch

    def _shift_right_t5(self, input_ids, init_token_id):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = init_token_id
        return shifted_input_ids

    def _encode_two(self, batch) -> Dict[str, torch.Tensor]:
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.data_args.max_source_length,
            max_target_length=self.data_args.max_target_length,
            padding=self.padding,  # TPU hack
            return_tensors="pt",
            **self.dataset_kwargs,
        )
        batch_encoding2 = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts2"] for x in batch],
            tgt_texts=[x["tgt_texts2"] for x in batch],
            max_length=self.data_args.max_source_length,
            max_target_length=self.data_args.max_target_length,
            padding=self.padding,  # TPU hack
            return_tensors="pt",
            **self.dataset_kwargs,
        )
        return batch_encoding.data, batch_encoding2.data

    def _encode(self, batch) -> Dict[str, torch.Tensor]:
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.data_args.max_source_length,
            max_target_length=self.data_args.max_target_length,
            padding=self.padding,  # TPU hack
            return_tensors="pt",
            **self.dataset_kwargs,
        )
        return batch_encoding.data
    
    def _bert_encode(self, batch):
        inputs = self.tokenizer(
            [x["src_texts"] for x in batch],
            truncation=True,
            max_length=self.data_args.max_source_length,
            padding=self.padding,  # TPU hack
            return_tensors="pt",
            **self.dataset_kwargs,
        )
        outputs = self.tokenizer(
            [x["tgt_texts"] for x in batch],
            truncation=True,
            max_length=self.data_args.max_target_length,
            padding=self.padding,  # TPU hack
            return_tensors="pt",
            **self.dataset_kwargs,
        )

        labels = outputs.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        output_batch = {
            "input_ids" : inputs.input_ids,
            "attention_mask" : inputs.attention_mask,
            "decoder_input_ids": outputs.input_ids,
            "decoder_attention_mask": outputs.attention_mask,
            "labels": labels
        }

        return output_batch

class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size, shuffle=True):
        self.data, self.bs, self.shuffle = data, batch_size, shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))


def sortish_sampler_indices(data: List, bs: int, shuffle=True) -> np.array:
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    if not shuffle:
        return np.argsort(np.array(data) * -1)

    def key_fn(i):
        return data[i]

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx


class DistributedSortishSampler(Sampler):
    """Copied from torch DistributedSampler"""

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, add_extra_examples=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_examples:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(dataset)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_examples
        self.shuffle = shuffle

    def __iter__(self) -> Iterable:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        sortish_data = [self.dataset.src_lens[i] for i in self.available_indices]
        sortish_indices = sortish_sampler_indices(sortish_data, self.batch_size, shuffle=self.shuffle)
        indices = [self.available_indices[i] for i in sortish_indices]
        assert len(indices) == self.num_samples
        return iter(indices)

    @cached_property
    def available_indices(self) -> np.array:
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        available_indices = indices[self.rank : self.total_size : self.num_replicas]
        return available_indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


logger = getLogger(__name__)


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"setting model.config to task specific params for {task}:\n {pars}")
        logger.info("note: command line args may override some of these")
        model.config.update(pars)


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, sort_keys=True, **json_dump_kwargs)


def load_json(path):
    with open(path) as f:
        return json.load(f)


ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]


def extract_rouge_mid_statistics(dct):
    new_dict = {}
    for k1, v1 in dct.items():
        mid = v1.mid
        new_dict[k1] = {stat: round(getattr(mid, stat), 4) for stat in ["precision", "recall", "fmeasure"]}
    return new_dict


def calculate_rouge(
    pred_lns: List[str],
    tgt_lns: List[str],
    use_stemmer=True,
    rouge_keys=ROUGE_KEYS,
    return_precision_and_recall=False,
    bootstrap_aggregation=True,
    newline_sep=True,
    rouge_lang=None,
) -> Dict:
    """Calculate rouge using rouge_scorer package.

    Args:
        pred_lns: list of summaries generated by model
        tgt_lns: list of groundtruth summaries (e.g. contents of val.target)
        use_stemmer:  Bool indicating whether Porter stemmer should be used to
        strip word suffixes to improve matching.
        rouge_keys:  which metrics to compute, defaults to rouge1, rouge2, rougeL, rougeLsum
        return_precision_and_recall: (False) whether to also return precision and recall.
        bootstrap_aggregation: whether to do the typical bootstrap resampling of scores. Defaults to True, if False
            this function returns a collections.defaultdict[metric: list of values for each observation for each subscore]``
        newline_sep:(default=True) whether to add newline between sentences. This is essential for calculation rougeL
        on multi sentence summaries (CNN/DM dataset).

    Returns:
         Dict[score: value] if aggregate else defaultdict(list) keyed by rouge_keys

    """
    logger.info("Rouge lang: " + str(rouge_lang))
    scorer = rouge_scorer.RougeScorer(
        rouge_keys, lang=rouge_lang,
        use_stemmer=use_stemmer
    )
    aggregator = scoring.BootstrapAggregator()
    for pred, tgt in zip(tgt_lns, pred_lns):
        # rougeLsum expects "\n" separated sentences within a summary
        if newline_sep:
            pred = add_newline_to_end_of_each_sentence(pred)
            tgt = add_newline_to_end_of_each_sentence(tgt)
        scores = scorer.score(pred, tgt)
        aggregator.add_scores(scores)

    if bootstrap_aggregation:
        result = aggregator.aggregate()
        if return_precision_and_recall:
            return extract_rouge_mid_statistics(result)  # here we return dict
        else:
            return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    else:
        return aggregator._scores  # here we return defaultdict(list)


# Utilities for freezing parameters and checking whether they are frozen


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5" or model_type == "mt5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"


def parse_numeric_n_bool_cl_kwargs(unparsed_args: List[str]) -> Dict[str, Union[int, float, bool]]:
    """
    Parse an argv list of unspecified command line args to a dict.
    Assumes all values are either numeric or boolean in the form of true/false.
    """
    result = {}
    assert len(unparsed_args) % 2 == 0, f"got odd number of unparsed args: {unparsed_args}"
    num_pairs = len(unparsed_args) // 2
    for pair_num in range(num_pairs):
        i = 2 * pair_num
        assert unparsed_args[i].startswith("--")
        if unparsed_args[i + 1].lower() == "true":
            value = True
        elif unparsed_args[i + 1].lower() == "false":
            value = False
        else:
            try:
                value = int(unparsed_args[i + 1])
            except ValueError:
                value = float(unparsed_args[i + 1])  # this can raise another informative ValueError

        result[unparsed_args[i][2:]] = value
    return result


def write_txt_file(ordered_tgt, path):
    f = Path(path).open("w")
    for ln in ordered_tgt:
        f.write(ln + "\n")
        f.flush()


def check_output_dir(args, expected_items=0):
    """
    Checks whether to bail out if output_dir already exists and has more than expected_items in it

    `args`: needs to have the following attributes of `args`:
      - output_dir
      - do_train
      - overwrite_output_dir

    `expected_items`: normally 0 (default) - i.e. empty dir, but in some cases a few files are expected (e.g. recovery from OOM)
    """
    if (
        os.path.exists(args.output_dir)
        and len(os.listdir(args.output_dir)) > expected_items
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and "
            f"has {len(os.listdir(args.output_dir))} items in it (expected {expected_items} items). "
            "Use --overwrite_output_dir to overcome."
        )

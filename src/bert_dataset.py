from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class BertDataset(Dataset):
    def __init__(
        self,
        clean_texts: List[str],
        corrupted_texts: List[str],
        tokenizer: PreTrainedTokenizer,
        *,
        labels: Optional[List[int]] = None,
        max_len: int = 512,
        mode: str = 'train',
        pad_first: bool = True,
    ) -> None:
        if len(clean_texts) != len(corrupted_texts):
            raise ValueError('found length mismatch between clean and corrupted texts')

        if labels is None and mode == 'train':
            raise ValueError('you must provide label when using train mode')

        if labels is not None and len(labels) != len(corrupted_texts):
            raise ValueError('found length mismatch between labels and texts')

        self._tokenizer = tokenizer
        self._clean_texts = clean_texts
        self._corrupted_texts = corrupted_texts
        self._labels = labels
        self._max_len = max_len
        self._pad_first = pad_first
        self._mode = mode

    def __len__(
        self,
    ) -> int:
        return len(self._clean_texts)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, torch.Tensor]:
        output = self._get_input_ids(idx)

        if self._mode == 'train':
            output['output_ids'] = self._get_output_ids(idx)
            output['label'] = self._get_labels(idx)

        return output

    def _pad(
        self,
        data: List[Any],
    ) -> List[Any]:
        pad_len = self._max_len - len(data)

        if self._pad_first:
            return [0 for _ in range(pad_len)] + data
        else:
            return data + [0 for _ in range(pad_len)]

    def _get_input_ids(
        self,
        idx: int,
    ) -> Dict[str, torch.Tensor]:
        input_ids = self._corrupted_texts[idx]
        input_ids = ['[CLS]'] + list(input_ids)[:min(len(input_ids), self._max_len - 2)] + ['[SEP]']
        input_ids = self._tokenizer.convert_tokens_to_ids(input_ids)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        return {
            'input_ids': torch.Tensor(self._pad(input_ids)),
            'input_mask': torch.Tensor(self._pad(input_mask)),
            'segment_ids': torch.Tensor(self._pad(segment_ids)),
        }

    def _get_labels(
        self,
        idx: int,
    ) -> torch.Tensor:
        label = self._labels[idx]
        label = [int(x) for x in label if x != ' ']
        label = [0] + label[:min(len(label), self._max_len - 2)] + [0]

        return torch.Tensor(self._pad(label))

    def _get_output_ids(
        self,
        idx: int,
    ) -> torch.Tensor:
        output_ids = self._clean_texts[idx]
        output_ids = ['[CLS]'] + list(output_ids)[:min(len(output_ids), self._max_len - 2)] + ['[SEP]']
        output_ids = self._tokenizer.convert_tokens_to_ids(output_ids)

        return torch.Tensor(self._pad(output_ids))

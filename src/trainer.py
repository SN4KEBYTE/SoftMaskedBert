from typing import List

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from src.models.soft_masked_bert import SoftMaskedBert
from src.optim_schedule import ScheduledOptim


class SoftMaskedBertTrainer:
    def __init__(
        self,
        bert: BertModel,
        tokenizer: BertTokenizer,
        *,
        gru_hidden_size: int = 256,
        gru_n_layers: int = 1,
        lr: float = 2e-5,
        gamma: float = 0.8,
        weight_decay: float = 0.01,
        warmup_steps: int = 10000,
        log_freq: int = 1000,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        self.tokenizer = tokenizer
        self.device = device
        self.gamma = gamma
        self.log_freq = log_freq

        self.model = SoftMaskedBert(
            bert,
            self.tokenizer.mask_token_id,
            gru_hidden_size,
            gru_n_layers,
            self.device,
        ).to(self.device)
        self.optim_schedule = ScheduledOptim(
            Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            ),
            gru_hidden_size,
            n_warmup_steps=warmup_steps,
        )
        self.criterion_c = nn.NLLLoss()
        self.criterion_d = nn.BCELoss()

    def train(
        self,
        train_data: DataLoader,
        epoch: int,
    ) -> float:
        self.model.train()

        return self._iteration(
            epoch,
            train_data,
        )

    def evaluate(
        self,
        val_data: DataLoader,
        epoch: int,
    ) -> float:
        self.model.eval()

        return self._iteration(
            epoch,
            val_data,
            train=False,
        )

    def inference(
        self,
        data_loader: DataLoader,
    ) -> List[str]:
        self.model.eval()

        output = []

        for data in tqdm(data_loader, total=len(data_loader),):
            data = {key: value.to(self.device) for key, value in data.items()}

            out, prob = self.model(
                data['input_ids'],
                data['input_mask'],
                data['segment_ids'],
            )
            output.extend(out.argmax(dim=-1).cpu().numpy().tolist())

        return [''.join(self.tokenizer.convert_ids_to_tokens(x)) for x in output]

    def _iteration(
        self,
        epoch: int,
        data_loader: DataLoader,
        *,
        train=True,
    ) -> float:
        stage = 'train' if train else 'val'

        data_loader = tqdm(
            enumerate(data_loader),
            desc=f'{stage}: epoch {epoch}',
            total=len(data_loader),
        )

        avg_loss = 0
        total_element = 0
        c_correct = 0
        d_correct = 0

        for idx, data in data_loader:
            data = {key: value.to(self.device) for key, value in data.items()}

            out, prob = self.model(
                data['input_ids'],
                data['input_mask'],
                data['segment_ids'],
            )
            prob = prob.reshape(
                -1,
                prob.shape[1],
            )
            loss_d = self.criterion_d(
                prob,
                data['label'].float(),
            )
            loss_c = self.criterion_c(
                out.transpose(
                    1,
                    2,
                ),
                data['output_ids'],
            )
            loss = self.gamma * loss_c + (1 - self.gamma) * loss_d

            if train:
                self.optim_schedule.zero_grad()
                loss.backward(retain_graph=True)
                self.optim_schedule.step_and_update_lr()

            out = out.argmax(dim=-1)
            c_correct += sum([out[i].equal(data['output_ids'][i]) for i in range(len(out))])
            prob = torch.round(prob).long()
            d_correct += sum([prob[i].equal(data['label'][i]) for i in range(len(prob))])

            avg_loss += loss.item()
            total_element += len(data)

            if idx % self.log_freq == 0:
                post_fix = {
                    'iter': idx,
                    'avg_loss': avg_loss / (idx + 1),
                    'd_acc': d_correct / total_element,
                    'c_acc': c_correct / total_element,
                }
                data_loader.write(str(post_fix))

        print(f'stage {stage} finished')
        print(f'avg loss = {avg_loss}')
        print(f'detection accuracy = {d_correct / total_element}')
        print(f'correction accuracy = {c_correct / total_element}')

        return avg_loss / len(data_loader)

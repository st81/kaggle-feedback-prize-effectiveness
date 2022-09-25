import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from transformers import AutoConfig, AutoModel

from config import Config
from utils.kaggle import HUGGING_FACE_MODEL_NAME_TO_KAGGLE_DATASET


class DenseCrossEntropy(nn.Module):
    def forward(self, x: torch.Tensor, target: torch.Tensor, weights=None):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=1)
        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class NLPAllclsTokenPooling(nn.Module):
    def __init__(self, dim: int, config: Config) -> None:
        super(NLPAllclsTokenPooling, self).__init__()

        self.dim = dim
        self.feat_mult = 1
        if config.dataset.group_discourse:
            self.feat_mult = 3

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        config: Config,
    ):
        if not config.dataset.group_discourse:
            raise NotImplementedError
        else:
            ret = []

            for j in range(x.shape[0]):
                # indexes which are equal to start_token_id
                idx0 = torch.where(
                    (input_ids[j] >= min(config._tokenizer_start_token_id))
                    & (input_ids[j] <= max(config._tokenizer_start_token_id))
                )[0]
                # indexes which are equal to end_token_id
                idx1 = torch.where(
                    (input_ids[j] >= min(config._tokenizer_end_token_id))
                    & (input_ids[j] <= max(config._tokenizer_end_token_id))
                )[0]
                xx = []
                for jj in range(len(idx0)):
                    # x where batch_idx = j and start_token_id. Shape: (self.backbone.config.hidden_size, )
                    xx0 = x[j, idx0[jj]]
                    # x where batch_idx = j and end_token_id. Shape: (self.backbone.config.hidden_size, )
                    xx1 = x[j, idx1[jj]]
                    # x.mean(dim=0) where batch_idx = j and between start_token_id and end_token_id. Shape: (self.backbone.config.hidden_size, )
                    xx2 = x[j, idx0[jj] + 1 : idx1[jj]].mean(dim=0)
                    # xxx.shape: (1, self.backbone.config.hidden_size * 3)
                    xxx = torch.cat([xx0, xx1, xx2]).unsqueeze(0)
                    xx.append(xxx)
                # xx.shape: (num_start_token_id, self.backbone.config.hidden_size * 3)
                xx = torch.cat(xx)
                ret.append(xx)

        return ret


class GeMText(nn.Module):
    def __init__(self, dim: int, config: Config, p: int = 3, eps: float = 1e-6) -> None:
        super(GeMText, self).__init__()
        self.dim = dim
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.feat_mult = 1

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        config: Config,
    ):
        attention_mask_expand = attention_mask.unsqueeze(-1).expand(x.shape)
        x = (x.clamp(min=self.eps) * attention_mask_expand).pow(self.p).sum(self.dim)
        ret = x / attention_mask_expand.sum(self.dim).clip(min=self.eps)
        ret = ret.pow(1 / self.p)
        return ret


class NLPPoolings:
    _poolings = {"All [CLS] token": NLPAllclsTokenPooling, "GeM": GeMText}

    @classmethod
    def get(cls, name):
        return cls._poolings.get(name)


class Net(nn.Module):
    def __init__(
        self,
        config: Config,
        is_test: bool = False,
        map_hugging_face_model_name_to_kaggle_dataset: bool = False,
    ) -> None:
        super(Net, self).__init__()

        self.config = config
        self.is_test = is_test

        backbone = (
            HUGGING_FACE_MODEL_NAME_TO_KAGGLE_DATASET[config.architecture.backbone]
            if map_hugging_face_model_name_to_kaggle_dataset
            else config.architecture.backbone
        )
        hf_config = AutoConfig.from_pretrained(backbone)
        if not is_test and self.config.architecture.custom_intermediate_dropout > 0:
            pass
        self.backbone = (
            AutoModel.from_pretrained(
                self.config.architecture.backbone, config=hf_config
            )
            if not is_test
            else AutoModel.from_config(hf_config)
        )

        if is_test:
            # I can not understand why this is needed,
            # but it is written in https://www.kaggle.com/code/ybabakhin/team-hydrogen-1st-place?scriptVersionId=104986984&cellId=7
            self.backbone.pooler = None

        if self.config.dataset.group_discourse:
            self.backbone.resize_token_embeddings(self.config._tokenizer_size)
        if not is_test:
            print(
                "Embedding size", self.backbone.embeddings.word_embeddings.weight.shape
            )

        self.pooling = NLPPoolings.get(self.config.architecture.pool)
        self.pooling = self.pooling(dim=1, config=config)

        dim = self.backbone.config.hidden_size * self.pooling.feat_mult

        self.dim = dim

        if not is_test and self.config.dataset.label_columns == ["discourse_type"]:
            raise NotImplementedError
        else:
            self.head = nn.Linear(dim, 3)

        if self.config.architecture.aux_type:
            raise NotImplementedError

        if not is_test:
            if self.config.training.loss_function == "CrossEntropy":
                self.loss_fn = DenseCrossEntropy()
            elif self.config.training.loss_function == "WeightedCrossEntropy":
                raise NotImplementedError
            elif self.config.training.loss_function == "FocalLoss":
                raise NotImplementedError

    def get_features(self, batch):
        attention_mask = batch["attention_mask"]
        input_ids = batch["input_ids"]

        x = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        x = self.pooling(x, attention_mask, input_ids, self.config)

        if self.config.dataset.group_discourse:
            if not self.is_test:
                # len(batch["target"]) = batch_size.
                batch["target"] = [
                    batch["target"][j][: len(x[j])] for j in range(len(x))
                ]
                batch["target"] = torch.cat(batch["target"])

                if self.training and self.config.architecture.aux_type:
                    raise NotImplementedError

            x = torch.cat(x)

        return x, batch

    def forward(self, batch, calculate_loss=True, is_test: bool = False):
        # Argument 'is_test' is exist for compatibility with 'models.feedback_model.Net'
        idx = int(torch.where(batch["attention_mask"] == 1)[1].max())
        idx += 1
        batch["attention_mask"] = batch["attention_mask"][:, :idx]
        batch["input_ids"] = batch["input_ids"][:, :idx]

        if self.training and self.config.training.mask_probability > 0:
            input_ids = batch["input_ids"].clone()
            special_mask = torch.ones_like(input_ids)
            special_mask[
                (input_ids == self.config._tokenizer_cls_token_id)
                | (input_ids == self.config._tokenizer_sep_token_id)
                | (input_ids == self.config._tokenizer_mask_token_id)
            ] = 0
            mask = (
                torch.bernoulli(
                    torch.full(input_ids.shape, self.config.training.mask_probability)
                )
                .to(input_ids.device)
                .bool()
                & special_mask
            ).bool()
            input_ids[mask] = self.config._tokenizer_mask_token_id
            batch["input_ids"] = input_ids.clone()

        x, batch = self.get_features(batch)
        if self.config.architecture.dropout > 0.0:
            raise NotImplementedError

        if self.config.architecture.wide_dropout > 0.0:
            raise NotImplementedError
        else:
            logits = self.head(x)

        if self.config.architecture.aux_type:
            raise NotImplementedError

        outputs = {}

        outputs["logits"] = logits

        if "target" in batch:
            outputs["target"] = batch["target"]

        if calculate_loss:
            targets = batch["target"]

            if self.training:
                outputs["loss"] = self.loss_fn(logits, targets)

                if self.config.architecture.aux_type:
                    raise NotImplementedError

        return outputs

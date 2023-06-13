import math
from typing import List, Optional, Tuple, Union, Any
from tqdm.auto import tqdm

import torch, random
import numpy as np
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, KLDivLoss

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import RobertaModel, RobertaTokenizer
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaClassificationHead
)

NELA_FEATS_CNT = 87

def set_random_seed(seed: int = 42) -> None:
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class RobertaForContrastiveClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, gamma = 0.5):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        self.gamma = gamma
        # self.alpha = alpha

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.shared = nn.Sequential(
            nn.Linear(config.hidden_size + NELA_FEATS_CNT, config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        self.classifier = RobertaClassificationHead(config)
        # self.pos = RobertaClassificationHead(config)
        # self.neg = RobertaClassificationHead(config)

        # for params in self.pos.parameters():
        #     params.requires_grad = False
        
        # for params in self.neg.parameters():
        #     params.requires_grad = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        nela_features: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        nela_features = nela_features.reshape(nela_features.shape[0], 1, nela_features.shape[1])
        # print(sequence_output.shape, nela_features.shape, nela_features.repeat(1, sequence_output.shape[1], 1).shape)

        sequence_output = torch.cat([sequence_output, nela_features.repeat(1, sequence_output.shape[1], 1)], dim = -1)

        shared_outs = self.shared(sequence_output)

        logits = self.classifier(shared_outs)

        # pos_logits = self.pos(sequence_output)
        # neg_logits = self.neg(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # kl_loss_fct = KLDivLoss(reduction = "batchmean")
            # n_loss = kl_loss_fct(
            #     F.log_softmax(logits.view(-1, self.num_labels), dim = 1),
            #     F.softmax(neg_logits.view(-1, self.num_labels), dim = 1)
            # )
            # p_loss = kl_loss_fct(
            #     F.log_softmax(logits.view(-1, self.num_labels), dim = 1),
            #     F.softmax(pos_logits.view(-1, self.num_labels), dim = 1)
            # )
            # kl_loss = torch.pow(n_loss, 1/self.gamma) - p_loss
            # print(logits[:, 0].view(-1, 1).shape)
            # kl_loss = kl_loss_fct(
            #     F.log_softmax(logits[:, 0].view(-1, 1), dim = 1),
            #     F.softmax(logits[:, 1].view(-1, 1), dim = 1)
            # ) + kl_loss_fct(
            #     F.log_softmax(logits[:, 0].view(-1, 1), dim = 1),
            #     F.softmax(logits[:, 2].view(-1, 1), dim = 1)
            # ) + kl_loss_fct(
            #     F.log_softmax(logits[:, 1].view(-1, 1), dim = 1),
            #     F.softmax(logits[:, 2].view(-1, 1), dim = 1)
            # )
            # ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # loss = ce_loss - self.gamma * kl_loss
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

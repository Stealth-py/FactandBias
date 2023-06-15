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

class MultiTaskRobertaForBiasFactualityCLS(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_tasks: int = 2, num_labels_per_task: List[int] = [3, 3]) -> None:
        super().__init__(config)
        self.num_labels = num_labels_per_task
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.num_tasks = num_tasks
        self.classifiers = nn.ModuleList([RobertaClassificationHead(config) for _ in range(self.num_tasks)])

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
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

        logits = []

        for i in range(self.num_tasks):
            logits.append(self.classifiers[i](sequence_output))

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            for i in range(self.num_tasks):
                loss += loss_fct(logits[i].view(-1, self.num_labels[i]), labels.view(-1, self.num_labels[i])[:, i])

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            all_logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

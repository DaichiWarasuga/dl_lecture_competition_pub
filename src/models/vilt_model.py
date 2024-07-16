import torch
import torch.nn as nn
from transformers import ViltForQuestionAnswering

class VQAModel_vilt(nn.Module):
    def init(self, num_labels, pretrained_model_name = "dandelin/vilt-b32-finetuned-vqa",
                 dropout_prob=0.1):
        super().init()
        self.num_labels = num_labels
        self.vilt = ViltForQuestionAnswering.from_pretrained(pretrained_model_name)
        # Update the classifier
        # self.vilt.config.num_labels = num_labels
        self.vilt.classifier = nn.Sequential(
            nn.Linear(in_features=768, out_features=1536, bias=True),
            nn.LayerNorm((1536,), eps=1e-05, elementwise_affine=True),
            nn.GELU(approximate='none'),
            nn.Dropout(dropout_prob),
            nn.Linear(in_features=1536, out_features=num_labels, bias=True)
        )
    def forward(self, input_ids, token_type_ids, pixel_values, attention_mask, pixel_mask):
        input_ids = input_ids.squeeze()
        pixel_values = pixel_values.squeeze(dim=1)
        pixel_mask = pixel_mask.squeeze(dim=1)
        outputs = self.vilt(input_ids=input_ids, token_type_ids=token_type_ids,
                            pixel_values=pixel_values, attention_mask=attention_mask,
                            pixel_mask=pixel_mask)
        return outputs
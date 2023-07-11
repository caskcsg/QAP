import math
from typing import Optional, List
import torch
from torch import nn
from src9.utils import padTensor
import os

class WrappedTransformerEncoder(nn.Module):
    def __init__(self, dim, num_layers, num_heads):
        super(WrappedTransformerEncoder, self).__init__()
        self.dim = dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=dim)

    def prepend_cls(self, inputs):
        #print(inputs.device)
        index = torch.LongTensor([0]).to(inputs.device)
        cls_emb = self.cls_emb(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, self.dim)
        outputs = torch.cat((cls_emb, inputs), dim=1)
        return outputs

    def forward(self, inputs: torch.Tensor, lens: Optional[List[int]] = None, get_cls: Optional[bool] = False, seq_len: Optional[int]=100, input_mask=None):
        #print(inputs.device)
        if lens is not None:
            max_len = max(lens)
            mask1 = []
            if input_mask==None:
                mask = []
                for l in  lens:
                    if l <= seq_len - 1:
                        mask.append([False] * (l + int(get_cls)) + [True] * (seq_len - l - 1))
                        mask1.append([1] * (l + int(get_cls)) + [0] * (seq_len - l - 1))
                    else:
                        mask.append([False] * (seq_len) )
                        mask1.append([1] * (seq_len) )
                mask = torch.tensor(mask).to(inputs.device)
                mask1 = torch.tensor(mask1).to(inputs.device)
            else:
                mask=input_mask
            inputs = list(inputs.split(lens, dim=0))
            inputs = [padTensor(inp, seq_len-1) for inp in inputs]
            inputs = torch.stack(inputs, dim=0)
            # mask = [([False] * (l + int(get_cls)) + [True] * (max_len - l)) for l in lens]
            # mask = torch.tensor(mask).to(device=inputs.device)
            # inputs = list(inputs.split(lens, dim=0))
            # inputs = [padTensor(inp, max_len) for inp in inputs]
            # inputs = torch.stack(inputs, dim=0)              
        else:
            mask = None
            mask1 = None
        #print(inputs.shape)
        if get_cls:
            inputs = self.prepend_cls(inputs)
        #print(inputs.shape)
        #exit(0)
        inputs = inputs.permute(1, 0, 2)
        # inputs = self.pos_encoder(inputs)
        inputs = self.encoder(src=inputs, src_key_padding_mask=mask) # (seq_len, bs, dim)

        # if get_cls:
        #     return inputs[0]
        return inputs.permute(1, 0, 2), mask, mask1
        #return inputs[1:].permute(1, 0, 2)


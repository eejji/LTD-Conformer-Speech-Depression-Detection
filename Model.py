import torch
from torch import Tensor
from typing import Tuple
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import Optional
import math

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import random
from sklearn.model_selection import train_test_split
import torch.optim as optim


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class Conv2dSubsampling(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        super(Conv2dSubsampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2), # 1, 512
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2), # 512, 512
            nn.ReLU()
        )

    def forward(self, inputs: Tensor,
                input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        # input : (batch, time, feature)
        outputs = self.sequential(inputs.unsqueeze(1)) 


        batch_size, channels, subsampled_length, sumsampled_dim = outputs.size()
        # (batch, 512, new time, new_dim)

        outputs = outputs.permute(0, 2, 1, 3)
        # 차원 재배치, (batch, new time, 512, new_dim)

        outputs = outputs.contiguous().view(batch_size, subsampled_length, channels * sumsampled_dim)
        # (batch, 512, 512 * new_dim)

        outputs_lengths = input_lengths >> 2
        outputs_lengths -= 1

        # output : (batch, 512, 512*new_dim)
        # outputs_lengths : (151 /4 -= 1)
        return outputs, outputs_lengths


class Linear(nn.Module):
    def __init__(self,
                 in_feature: int,
                 out_feature: int,
                 bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_feature, out_feature, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


# Conformerencoder 정의
class ConformerEncoder(nn.Module):
    def __init__(self,
                 input_dim: int = 80,
                 encoder_dim: int = 512,
                 num_layers: int = 17,
                 num_attention_heads: int = 8,
                 feed_forward_expansion_factor: int = 4,
                 conv_expansion_factor: int = 2,
                 input_dropout_rate: float = 0.1,
                 feed_forward_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 conv_dropout_rate: float = 0.1,
                 conv_kernel_size: int = 31,
                 half_step_residual: bool = True):
        super(ConformerEncoder, self).__init__()

        self.conv_subsample = Conv2dSubsampling(in_channels=1, out_channles=encoder_dim)
        self.inputs_projection = nn.Sequential(
            Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim),

            nn.Dropout(p=input_dropout_rate)
        )

        self.layers = nn.ModuleList([ConformerBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_rate=feed_forward_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            conv_dropout_rate=conv_dropout_rate,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        ) for _ in range(num_layers)])

    def count_parameters(self) -> int:
        return sum([p.numel() for p in self.parameters()])


    def update_dropout(self, dropout_rate: float) -> None:
        for name, child in self.name_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_rate

    def forward(self,
                inputs: Tensor,
                input_lengths: Tensor) -> Tuple[Tensor, Tensor]:


        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        outputs = self.inputs_projection(outputs)

        # conformer block process
        for layer in self.layers:
            # layers 수 만큼 block stack
            outputs = layer(outputs)

        return outputs, output_lengths


# Conformer block 정의
class ConformerBlock(nn.Module):
    def __init__(self,
                 encoder_dim: int = 512,
                 num_attention_heads: int = 8,
                 feed_forward_expansion_factor: int = 4,
                 conv_expansion_factor: int = 2,
                 feed_forward_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 conv_dropout_rate: float = 0.1,
                 conv_kernel_size: int = 31,
                 half_step_residual: bool = True):

        super(ConformerBlock, self).__init__()

        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.sequential = nn.Sequential(
            # Feed Froward network
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_rate=feed_forward_dropout_rate
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            # Mult-Head Self Attention Module
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_rate=attention_dropout_rate
                )
            ),
            # Convolution Module
            ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    dropout_rate=conv_dropout_rate
                )
            ),
            # Feed Forward network
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_rate=feed_forward_dropout_rate
                ),
                module_factor=self.feed_forward_residual_factor
            ),
            # LayerNorm
            nn.LayerNorm(encoder_dim)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)


class Conformer(nn.Module):
    """
    Args:
        num_classes (int): Number of classification classes
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of conformer encoder
        num_encoder_layers (int, optional): Number of conformer blocks
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths

    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by conformer.
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(self,
                 num_classes: int,
                 input_dim: int = 80,
                 encoder_dim: int = 512,
                 num_encoder_layers: int = 17,
                 num_attention_heads: int = 8,
                 feed_forward_expansion_factor: int = 4,
                 conv_expansion_factor: int = 2,
                 input_dropout_rate: float = 0.1,
                 feed_forward_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 conv_dropout_rate: float = 0.1,
                 conv_kernel_size: int = 31,
                 half_step_residual: bool = True
                ) -> None:

        super(Conformer, self).__init__()
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_rate=input_dropout_rate,
            feed_forward_dropout_rate=feed_forward_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            conv_dropout_rate=conv_dropout_rate,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual
        )
        self.fc = Linear(encoder_dim, num_classes, bias=False)

    def count_parameters(self) -> int:
        return self.encoder.count_parameters()

    def update_dropout(self, dropout_rate) -> None:
        self.encoder.update_dropout(dropout_rate)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        encoder_outputs, encoder_outputs_lengths = self.encoder(inputs, input_lengths)
        outputs = self.fc(encoder_outputs)
        outputs = torch.mean(outputs, dim=1)
        outputs = nn.functional.log_softmax(outputs, dim=-1)

        return outputs, encoder_outputs_lengths

class ResidualConnectionModule(nn.Module):
    def __init__(self, module: nn.Module,
                 module_factor: float = 1.0,
                 input_factor: float = 1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor) -> Tensor:
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)
        # self.module(inputs) -> inputs을 module을 거처 반환
        # self.module_factor 변환된 결과에 module_factor를 곱하여 가중치 적용
        # input_factor : inputs에 곱해지는 input 가중치


class FeedForwardModule(nn.Module):
    '''
    encoder_dim
    expansion_factor = FFN의 차원
    dropout rate

    inputs : (batch, time, dim)
    '''

    def __init__(self,
                 encoder_dim: int = 512,
                 expansion_factor: int = 4,
                 dropout_rate: float = 0.1
                 ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            # (dim -> dim * expansion_factor)
            Swish(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            # (dim * expansion -> dim)
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # (batch, time, dim)
        return self.sequential(inputs)


class MultiHeadedSelfAttentionModule(nn.Module):
    """
        Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """

    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size = inputs.size(0)
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)


class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module.
    Args:
        d_model: Embedding dimension.
        max_len: Maximum input length.
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor) -> None:
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * - (math.log(100000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : Input Tensor Batch X Time X Channel
        Returns:
            torch.Tensor: Encoded tensor Batch X Time X Channel
        """
        self.extend_pe(x)
        pos_emb = self.pe[:, self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1)]
        return pos_emb


class RelativeMultiHeadAttention(nn.Module):
    """
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(self,
                 d_model: int = 512,
                 num_heads: int = 16,
                 dropout_rate: float = 0.1):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        # Multi-head Attention의 Linear 부분
        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_rate)

        # Relative Positional Embedding에 사용되는 부분
        # 두 개의 Learnable Bias parameter 추가
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                pos_embedding: Tensor,
                mask: Optional[Tensor] = None,) -> Tensor:

        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        # torch.permute : (0, 1, 2, 3) -> (0, 2, 1, 3) 순으로 형태를 변경
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1,2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()

        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)

        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

        return pos_score


class ConformerConvModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int = 31,
                 expansion_factor: int = 2,
                 dropout_rate: float = 0.1) -> None:
        super(ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding "
        assert expansion_factor == 2, "currently, Only supports expansion_factor 2 "

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            PointwiseConv1d(in_channels, in_channels * expansion_factor,
                            stride=1, padding=0, bias=True),
            GLU(dim=1),
            DepthwiseConv1d(in_channels, in_channels, kernel_size,
                            stride=1, padding=(kernel_size-1) // 2),
            Swish(),
            PointwiseConv1d(in_channels, in_channels, stride=1,
                            padding=0, bias=True),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs).transpose(1, 2)


class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class PointwiseConv1d(nn.Module): # 1d convolution 이라고 생각
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 padding: int = 0,
                 bias: bool = True
                 ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,  # Pointwise Convolution
                              stride=stride,
                              padding=padding,
                              bias=bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class DepthwiseConv1d(nn.Module): # 차후에 dilation을 추가할 예정
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 bias: bool = False) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              groups=in_channels,
                              stride=stride,
                              padding=padding,
                              bias=bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class LTD_DepthwiseConv1d(nn.Module): 
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 2,
                 padding: int = 0,
                 bias: bool = False) -> None:
        super(LTD_DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out 채널은 in_channle의 배수여야한다"
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              groups=in_channels,
                              dilation=dilation,
                              stride=stride,
                              padding=padding,
                              bias=bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class LTD_ConformerConvModule(nn.Module):
    """
    D-Conformer에서 사용할 Dilated CNN 기반 Convolution Module
    kernel_size와 dilation을 이용해 receptive field 확장
    """

    def __init__(self,
                 in_channels: int,
                 kernel_size: int = 31,
                 dilation: int = 2,
                 expansion_factor: int = 2,
                 dropout_rate: float = 0.1) -> None:
        super(LTD_ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be odd"
        assert expansion_factor == 2, "only supports expansion_factor=2 for now"

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),  # (batch, time, dim) -> (batch, dim, time)
            PointwiseConv1d(in_channels, in_channels * expansion_factor,
                            stride=1, padding=0, bias=True),
            GLU(dim=1),
            # DilatedDepthwiseConv1d 적용
            LTD_DepthwiseConv1d(in_channels,
                                in_channels,
                                kernel_size=kernel_size,
                                dilation=dilation,
                                stride=1,
                                padding=((kernel_size-1)//2)*dilation),
            Swish(),
            nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1,
                      padding=0, bias=True),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, time, dim) -> (batch, time, dim)
        return self.sequential(x).transpose(1, 2)


class LTD_ConformerBlock(nn.Module):
    def __init__(self,
                 encoder_dim: int = 512,
                 num_attention_heads: int = 8,
                 feed_forward_expansion_factor: int = 4,
                 conv_expansion_factor: int = 2,
                 feed_forward_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 conv_dropout_rate: float = 0.1,
                 conv_kernel_size: int = 31,
                 dilation: int = 2,
                 half_step_residual: bool = True):
        super().__init__()
        if half_step_residual:
            self.ff_res_factor = 0.5
        else:
            self.ff_res_factor = 1.0

        self.sequential = nn.Sequential(
            # 1) 첫 번째 Feed Forward
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_rate=feed_forward_dropout_rate
                ),
                module_factor=self.ff_res_factor
            ),
            # 2) Multi-Head Self Attention
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_rate=attention_dropout_rate
                )
            ),
            # 3) Dilated Convolution Module
            ResidualConnectionModule(
                module=LTD_ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    dilation=dilation,
                    dropout_rate=conv_dropout_rate
                )
            ),
            # 4) 두 번째 Feed Forward
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_rate=feed_forward_dropout_rate
                ),
                module_factor=self.ff_res_factor
            ),
            # 5) LayerNorm
            nn.LayerNorm(encoder_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


class LTD_LongTermModule(nn.Module):
    """
    장기 패턴 학습을 위한 GRU + LayerNorm 모듈 예시
    hidden_size는 D-Conformer의 encoder_dim과 동일하게 설정
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super(LTD_LongTermModule, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=False,
                           dropout=dropout if num_layers > 1 else 0.0,
                           bidirectional=False)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, time, dim=hidden_size)
        return: (batch, time, dim)
        """
        out, _ = self.gru(x)  # (batch, time, hidden_size)
        out = self.layer_norm(out)
        return out

class LTD_ConformerEncoder(nn.Module):
    def __init__(self,
                 input_dim: int = 80,
                 encoder_dim: int = 512,
                 num_layers: int = 17,
                 num_attention_heads: int = 8,
                 feed_forward_expansion_factor: int = 4,
                 conv_expansion_factor: int = 2,
                 input_dropout_rate: float = 0.1,
                 feed_forward_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 conv_dropout_rate: float = 0.1,
                 conv_kernel_size: int = 31,
                 dilation: int = 2,
                 half_step_residual: bool = True,
                 use_longterm: bool = True,  # Long-term Module 사용 여부
                 longterm_num_layers: int = 2):
        super(LTD_ConformerEncoder, self).__init__()
        self.use_longterm = use_longterm

        # 2D Subsampling
        self.conv_subsample = Conv2dSubsampling(in_channels=1, out_channels=encoder_dim)
        self.input_projection = nn.Sequential(
            Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim),
            nn.Dropout(p=input_dropout_rate)
        )

        # DConformer Blocks
        self.layers = nn.ModuleList([
            LTD_ConformerBlock(
                encoder_dim=encoder_dim,
                num_attention_heads=num_attention_heads,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                feed_forward_dropout_rate=feed_forward_dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                conv_dropout_rate=conv_dropout_rate,
                conv_kernel_size=conv_kernel_size,
                dilation=dilation,
                half_step_residual=half_step_residual
            ) for _ in range(num_layers)
        ])
        self.fusion_module = HybridFusion(encoder_dim)
        # Long-term Module (GRU + LayerNorm)
        if use_longterm:
            self.longterm = LTD_LongTermModule(input_size=encoder_dim,
                                               hidden_size=encoder_dim,
                                               num_layers=longterm_num_layers,
                                               dropout=feed_forward_dropout_rate)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) Conformer encoder (conv2dsubsample -> Linear -> dropout)

        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
    
        outputs = self.input_projection(outputs)  # (batch, time, encoder_dim)

        if self.use_longterm:
            longterm_out = self.longterm(outputs)
            # inptus을 바로 long-term module에 입력 (batch, number of window, feature)

        # 2) D-Conformer Blocks
        for layer in self.layers:
            outputs = layer(outputs)  # (batch, time, encoder_dim)

        total_outputs = outputs * longterm_out

        return total_outputs, output_lengths


class LTD_Conformer(nn.Module):
    def __init__(self,
                 num_classes: int,
                 input_dim: int = 80,
                 encoder_dim: int = 512,
                 num_layers: int = 17,
                 num_attention_heads: int = 8,
                 feed_forward_expansion_factor: int = 4,
                 conv_expansion_factor: int = 2,
                 input_dropout_rate: float = 0.1,
                 feed_forward_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 conv_dropout_rate: float = 0.1,
                 conv_kernel_size: int = 31,
                 dilation: int = 2,
                 half_step_residual: bool = True,
                 use_longterm: bool = True,
                 longterm_num_layers: int = 2):
        super().__init__()
        self.encoder = LTD_ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_rate=input_dropout_rate,
            feed_forward_dropout_rate=feed_forward_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            conv_dropout_rate=conv_dropout_rate,
            conv_kernel_size=conv_kernel_size,
            dilation=dilation,
            half_step_residual=half_step_residual,
            use_longterm=use_longterm,
            longterm_num_layers=longterm_num_layers
        )
        self.fc = Linear(encoder_dim, num_classes, bias=False)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) Encoder
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        #2) Pooling (예: mean pooling) 후 분류
        outputs = torch.mean(encoder_outputs, dim=1)  # (batch, encoder_dim)
        outputs = self.fc(outputs)                    # (batch, num_classes)
        outputs = F.log_softmax(outputs, dim=-1)

        return outputs
from torch import Tensor
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
# from utils import create_mask

import torchvision
from torch.nn.utils.rnn import pad_sequence
#import pytorchvideo.models.x3d as x3d
import utils as utils

""" PyTorch MBART model."""
from transformers import MBartForConditionalGeneration, MBartPreTrainedModel, MBartModel, MBartConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.models.mbart.modeling_mbart import shift_tokens_right

from transformers.models.mbart.modeling_mbart import MBartLearnedPositionalEmbedding, MBartEncoderLayer

from collections import OrderedDict

import torch
from typing import Optional

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    手动实现掩码扩展功能，适配各种transformers版本
    功能：将二维掩码 (bsz, src_len) 扩展为四维 (bsz, 1, tgt_len, src_len)，用于注意力计算
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len  
    
  
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask  
    
 
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)
import copy
import math
import random
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

# global definition
from definition import *

from hpman.m import _
from pathlib import Path


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


def make_resnet(name='resnet18'):
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    else:
        raise Exception('There are no supported resnet model {}.'.format(_('resnet')))

    inchannel = model.fc.in_features
    model.fc = nn.Identity()
    return model

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.resnet = make_resnet(name='resnet18')

    def forward(self, x, lengths):
        x = self.resnet(x)
        x_batch = []
        start = 0
        for length in lengths:
            end = start + length
            x_batch.append(x[start:end])
            start = end
        x = pad_sequence(x_batch,padding_value=PAD_IDX,batch_first=True)
        return x
  
class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.temporal_conv(x.permute(0,2,1))
        return x.permute(0,2,1)
    
def make_head(inplanes, planes, head_type):
    if head_type == 'linear':
        return nn.Linear(inplanes, planes, bias=False)
    else:
        return nn.Identity()

class TextCLIP(nn.Module):
    def __init__(self, config=None, inplanes=1024, planes=1024, head_type='identy'):
        super(TextCLIP, self).__init__()

        self.model_txt = MBartForConditionalGeneration.from_pretrained(config['model']['transformer']).get_encoder() 

        self.lm_head = make_head(inplanes, planes, head_type)

    def forward(self, tgt_input):
        txt_logits = self.model_txt(input_ids=tgt_input['input_ids'].cuda(), attention_mask=tgt_input['attention_mask'].cuda())[0]
        output = txt_logits[torch.arange(txt_logits.shape[0]), tgt_input['input_ids'].argmax(dim=-1)]
        return self.lm_head(output), txt_logits

class ImageCLIP(nn.Module):
    def __init__(self, config, inplanes=1024, planes=1024, head_type='linear') :
        super(ImageCLIP, self).__init__()
        self.config = config
        self.model =  FeatureExtracter() 
        
        self.trans_encoder = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_encoder()
        self.cls_token = nn.Parameter(torch.randn(1, 1, inplanes))

        self.lm_head = make_head(inplanes, planes, head_type)
        
    def forward(self, src_input):
       
        x = self.model(src_input['input_ids'].cuda(), src_input['src_length_batch']) # [b, n, c]
        attention_mask = src_input['attention_mask']

        B, N, C = x.shape
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=B)
        x = torch.cat((cls_token, x), dim=1)
        attention_mask = F.pad(attention_mask.flatten(1), (1, 0), value=1.)  # [b, 64] --> [b, 65]

        outs = self.trans_encoder(inputs_embeds=x, attention_mask=attention_mask.cuda(), return_dict=True)
        last_hidden_state = outs['last_hidden_state']
        output = self.lm_head(last_hidden_state[:, 0, :])
        return output
    
class TextCLIP_Prompt(nn.Module):
    def __init__(self, config=None, inplanes=1024, planes=1024, head_type='identy', num_prompts=10):
        """
        Added num_prompts parameter to control the number of prompt vectors.
        """
        super(TextCLIP_Prompt, self).__init__()

        self.model_txt = MBartForConditionalGeneration.from_pretrained(config['model']['transformer']).get_encoder() 
 
        # Store the number of prompts as an attribute
        self.num_prompts = num_prompts
        
        # Create the learnable soft prompts as a parameter
        # Shape: [1, num_prompts, embedding_dimension]
        self.soft_prompts = nn.Parameter(torch.randn(1, self.num_prompts, inplanes))

        self.lm_head = make_head(inplanes, planes, head_type)

    def forward(self, tgt_input):
        input_ids = tgt_input['input_ids'].cuda()
        attention_mask = tgt_input['attention_mask'].cuda()
        B = input_ids.shape[0]

        # 1. Get the standard token embeddings from the model's embedding layer
        token_embeds = self.model_txt.embed_tokens(input_ids)

        # 2. Expand the soft prompts to match the batch size
        # Shape: [1, num_prompts, C] -> [B, num_prompts, C]
        prompts = self.soft_prompts.expand(B, -1, -1)

        # 3. Prepend the prompt embeddings to the token embeddings
        # The new input sequence is [prompts, token_embeddings]
        inputs_embeds = torch.cat((prompts, token_embeds), dim=1)

        # 4. Create an attention mask for the prompts (all ones)
        prompt_mask = torch.ones(B, self.num_prompts, device=attention_mask.device)
        # Concatenate the prompt mask with the original attention mask
        new_attention_mask = torch.cat((prompt_mask, attention_mask), dim=1)

        # 5. Pass the combined embeddings and new mask to the encoder
        # Note: We use `inputs_embeds` instead of `input_ids`
        txt_logits = self.model_txt(inputs_embeds=inputs_embeds, attention_mask=new_attention_mask)[0]

        # 6. Adjust the index for the last token to account for the prepended prompts
        # The original indices are now shifted by `num_prompts`
        last_token_indices = input_ids.argmax(dim=-1) + self.num_prompts

        # Gather the hidden state of the last token using the *adjusted* indices
        output = txt_logits[torch.arange(txt_logits.shape[0]), last_token_indices]
        
        return self.lm_head(output), txt_logits


class ImageCLIP_Prompt(nn.Module):
    def __init__(self, config, inplanes=1024, planes=1024, head_type='linear', num_prompts=10):
        
        super(ImageCLIP_Prompt, self).__init__()
        self.config = config
        self.model = FeatureExtracter()

        self.trans_encoder = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_encoder()
        
        # 存储 prompt 的数量
        self.num_prompts = num_prompts
        # 创建可学习的 soft prompts，形状为 [1, num_prompts, hidden_size]
        self.soft_prompts = nn.Parameter(torch.randn(1, self.num_prompts, inplanes))

        self.cls_token = nn.Parameter(torch.randn(1, 1, inplanes))
        self.lm_head = make_head(inplanes, planes, head_type)

    def forward(self, src_input):
        x = self.model(src_input['input_ids'].cuda(), src_input['src_length_batch'])  # [b, n, c]
        attention_mask = src_input['attention_mask']

        B, N, C = x.shape

        prompts = repeat(self.soft_prompts, '() n d -> b n d', b=B)
        
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=B)

        x = torch.cat((prompts, cls_token, x), dim=1)
        
        total_new_tokens = self.num_prompts + 1
        attention_mask = F.pad(attention_mask.flatten(1), (total_new_tokens, 0), value=1.)

        outs = self.trans_encoder(inputs_embeds=x, attention_mask=attention_mask.cuda(), return_dict=True)
        last_hidden_state = outs['last_hidden_state']

        output = self.lm_head(last_hidden_state[:, self.num_prompts, :])
        
        
        return output

class Text_Decoder(nn.Module):
    def __init__(self, config):
        super(Text_Decoder, self).__init__()
        self.text_decoder = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_decoder()
        self.lm_head = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_output_embeddings()
        self.register_buffer("final_logits_bias", torch.zeros((1, MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).model.shared.num_embeddings)))

    
    def forward(self, tgt_input, masked_tgt_input, model_txt):
        with torch.no_grad():
            _, encoder_hidden_states = model_txt(masked_tgt_input)

        decoder_input_ids = shift_tokens_right(tgt_input['input_ids'].cuda(), self.text_decoder.config.pad_token_id)
        decoder_out = self.text_decoder(
                    input_ids = decoder_input_ids,
                    attention_mask = tgt_input['attention_mask'].cuda(),
                    encoder_hidden_states = encoder_hidden_states,
                    encoder_attention_mask = masked_tgt_input['attention_mask'].cuda(),
                    return_dict = True,
                    )
        lm_logits = self.lm_head(decoder_out[0]) + self.final_logits_bias

        return lm_logits
    
        
class SLRCLIP(nn.Module):
    def __init__(self, config, embed_dim=1024) :
        super(SLRCLIP, self).__init__()
        self.model_txt = TextCLIP(config, inplanes=embed_dim, planes=embed_dim)
        self.model_images = ImageCLIP(config, inplanes=embed_dim, planes=embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_model_txt(self):
        return self.model_txt
    
    @property
    def get_encoder_hidden_states(self):
        return self.encoder_hidden_states
    
    def forward(self, src_input, tgt_input):
        image_features = self.model_images(src_input)
        text_features, self.encoder_hidden_states = self.model_txt(tgt_input)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        ground_truth = torch.eye(logits_per_image.shape[0], device=logits_per_text.device, dtype=logits_per_image.dtype, requires_grad=False)

        return logits_per_image, logits_per_text, ground_truth
       
class SLRCLIP_Prompt(nn.Module):
    def __init__(self, config, embed_dim=1024) :
        super(SLRCLIP_Prompt, self).__init__()
        self.model_txt = TextCLIP_Prompt(config, inplanes=embed_dim, planes=embed_dim)
        self.model_images = ImageCLIP_Prompt(config, inplanes=embed_dim, planes=embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_model_txt(self):
        return self.model_txt
    
    @property
    def get_encoder_hidden_states(self):
        return self.encoder_hidden_states
    
    def forward(self, src_input, tgt_input):
        image_features = self.model_images(src_input)
        text_features, self.encoder_hidden_states = self.model_txt(tgt_input)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        ground_truth = torch.eye(logits_per_image.shape[0], device=logits_per_text.device, dtype=logits_per_image.dtype, requires_grad=False)

        return logits_per_image, logits_per_text, ground_truth

class FeatureExtracter(nn.Module):
    def __init__(self, frozen=False):
        super(FeatureExtracter, self).__init__()
        self.conv_2d = resnet() # InceptionI3d()
        self.conv_1d = TemporalConv(input_size=512, hidden_size=1024, conv_type=2)

        if frozen:
            for param in self.conv_2d.parameters():
                param.requires_grad = False

    def forward(self,
                src: Tensor,
                src_length_batch
                ):
        src = self.conv_2d(src,src_length_batch)
        src = self.conv_1d(src)

        return src

class V_encoder(nn.Module):
    def __init__(self,
                 emb_size,
                 feature_size,
                 config,
                 ):
        super(V_encoder, self).__init__()
        
        self.config = config

        self.src_emb = nn.Linear(feature_size, emb_size)
        modules = []
        modules.append(nn.BatchNorm1d(emb_size))
        modules.append(nn.ReLU(inplace=True))
        self.bn_ac = nn.Sequential(*modules)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,
                src: Tensor,
                ):
      
        src = self.src_emb(src)
        src = self.bn_ac(src.permute(0,2,1)).permute(0,2,1)

        return src

def config_decoder(config):
    from transformers import AutoConfig
    
    decoder_type = _('decoder_type', 'LD', choices=['LD', 'LLMD'])
    if decoder_type == 'LD':
        
        return MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder'], ignore_mismatched_sizes = True, config = AutoConfig.from_pretrained(Path(config['model']['visual_encoder'])/'config.json'))
    elif decoder_type == 'LLMD':
        return MBartForConditionalGeneration.from_pretrained(config['model']['transformer'], ignore_mismatched_sizes = True, config = AutoConfig.from_pretrained(Path(config['model']['transformer'])/'LLMD_config.json'))
    
class gloss_free_model(nn.Module):
    def __init__(self, config, args, embed_dim=1024, pretrain=None):
        super(gloss_free_model, self).__init__()
        self.config = config
        self.args = args

        self.backbone = FeatureExtracter(frozen=_('freeze_backbone', False))
        # self.mbart = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder'])
        self.mbart = config_decoder(config)
 
        if config['model']['sign_proj']:
            self.sign_emb = V_encoder(emb_size=embed_dim,feature_size=embed_dim, config = config)
            self.embed_scale = math.sqrt(embed_dim) if config['training']['scale_embedding'] else 1.0
        else:
            self.sign_emb = nn.Identity()
            self.embed_scale = 1.0
        
    def share_forward(self, src_input):
        
        frames_feature = self.backbone(src_input['input_ids'].cuda(), src_input['src_length_batch'])
        attention_mask = src_input['attention_mask']

        inputs_embeds = self.sign_emb(frames_feature)
        inputs_embeds = self.embed_scale * inputs_embeds

        return inputs_embeds, attention_mask

    def forward(self,src_input, tgt_input ):
        
        inputs_embeds, attention_mask = self.share_forward(src_input)

        out = self.mbart(inputs_embeds = inputs_embeds,
                    attention_mask = attention_mask.cuda(),
                    # decoder_input_ids = tgt_input['input_ids'].cuda(),
                    labels = tgt_input['input_ids'].cuda(),
                    decoder_attention_mask = tgt_input['attention_mask'].cuda(),
                    return_dict = True,
                    )
        return out['logits'],inputs_embeds
    

    def generate(self,src_input,max_new_tokens,num_beams,decoder_start_token_id ):
        inputs_embeds, attention_mask = self.share_forward(src_input)

        out = self.mbart.generate(inputs_embeds = inputs_embeds,
                    attention_mask = attention_mask.cuda(),max_new_tokens=max_new_tokens,num_beams = num_beams,
                                decoder_start_token_id=decoder_start_token_id
                            )
        return out


class gloss_free_model_Prompt(nn.Module):
    def __init__(self, config, args, embed_dim=1024, pretrain=None, num_prompts=50):
        super(gloss_free_model_Prompt, self).__init__()
        self.config = config
        self.args = args

        self.backbone = FeatureExtracter(frozen=config.get('freeze_backbone', False))
        self.mbart = config_decoder(config)
 
        if config['model']['sign_proj']:
            self.sign_emb = V_encoder(emb_size=embed_dim, feature_size=embed_dim, config=config)
            self.embed_scale = math.sqrt(embed_dim) if config['training']['scale_embedding'] else 1.0
        else:
            self.sign_emb = nn.Identity()
            self.embed_scale = 1.0
        
        self.num_prompts = num_prompts
        if self.num_prompts > 0:
        
            self.visual_soft_prompts = nn.Parameter(torch.randn(1, self.num_prompts, embed_dim))
            self.task_vector_vq = nn.Parameter(torch.randn(1, self.num_prompts, embed_dim))
            self.task_vector_vl = nn.Parameter(torch.randn(1, self.num_prompts, embed_dim))

            self.text_soft_prompts = nn.Parameter(torch.randn(1, self.num_prompts, embed_dim))
            self.task_vector_tq = nn.Parameter(torch.randn(1, self.num_prompts, embed_dim))
            self.task_vector_tl = nn.Parameter(torch.randn(1, self.num_prompts, embed_dim))
      
            self.text_prompt_mlp = nn.Sequential(
                nn.Linear(embed_dim, 32),
                nn.ReLU(),
                nn.Linear(32, embed_dim)
            )

        self._cached_updated_visual_prompts = None
        self._cached_updated_text_prompts = None

    def share_forward(self, src_input):
        
        frames_feature = self.backbone(src_input['input_ids'].cuda(), src_input['src_length_batch'])
        attention_mask = src_input['attention_mask']

        inputs_embeds = self.sign_emb(frames_feature)
        inputs_embeds = self.embed_scale * inputs_embeds

        if self.num_prompts > 0:
            B = inputs_embeds.shape[0]
            embed_dim = inputs_embeds.shape[-1]
            
            prompts_vq = (self.visual_soft_prompts + self.task_vector_vq).expand(B, -1, -1)
            prompts_vl = (self.visual_soft_prompts + self.task_vector_vl).expand(B, -1, -1)
            
            stacked_prompts = torch.stack((prompts_vq, prompts_vl), dim=2)
            combined_prompts = stacked_prompts.view(B, 2 * self.num_prompts, embed_dim)
            inputs_embeds = torch.cat((combined_prompts, inputs_embeds), dim=1)
            attention_mask = F.pad(attention_mask, (2 * self.num_prompts, 0), value=1.)

        return inputs_embeds, attention_mask

    def forward(self, src_input, tgt_input):

        inputs_embeds, attention_mask = self.share_forward(src_input)
        B, _, embed_dim = inputs_embeds.shape

        decoder_inputs_embeds = self.mbart.model.decoder.embed_tokens(tgt_input['input_ids'].cuda())
        decoder_attention_mask = tgt_input['attention_mask'].cuda()
        labels = tgt_input['input_ids'].cuda()

        if self.num_prompts > 0:
            processed_text_prompts = self.text_prompt_mlp(self.text_soft_prompts)
            prompts_tq = (processed_text_prompts + self.task_vector_tq).expand(B, -1, -1)
            prompts_tl = (processed_text_prompts + self.task_vector_tl).expand(B, -1, -1)

            stacked_text_prompts = torch.stack((prompts_tq, prompts_tl), dim=2)
            combined_text_prompts = stacked_text_prompts.view(B, 2 * self.num_prompts, embed_dim)
            decoder_inputs_embeds = torch.cat((combined_text_prompts, decoder_inputs_embeds), dim=1)
            decoder_attention_mask = F.pad(decoder_attention_mask, (2 * self.num_prompts, 0), value=1.)
            labels = F.pad(labels, (2 * self.num_prompts, 0), value=-100)

        out = self.mbart(inputs_embeds=inputs_embeds,
                         attention_mask=attention_mask,
                         decoder_inputs_embeds=decoder_inputs_embeds,
                         decoder_attention_mask=decoder_attention_mask,
                         labels=labels,
                         return_dict=True,
                        )
        return out['logits'], inputs_embeds

    def _decorrelate_and_reconstruct(self, T_Q, T_L):
      
        T_Q = T_Q.squeeze(0)
        T_L = T_L.squeeze(0)

        U_Q, S_Q, Vh_Q = torch.linalg.svd(T_Q, full_matrices=False)
        U_L, S_L, Vh_L = torch.linalg.svd(T_L, full_matrices=False)

        U_cat = torch.cat((U_Q, U_L), dim=1)
        Sigma_cat = torch.block_diag(torch.diag(S_Q), torch.diag(S_L))
        
        V_cat = torch.cat((Vh_Q.mH, Vh_L.mH), dim=1)

        P_U, _, Qh_U = torch.linalg.svd(U_cat, full_matrices=False)
        U_perp = P_U @ Qh_U

        P_V, _, Qh_V = torch.linalg.svd(V_cat, full_matrices=False)
        V_perp = P_V @ Qh_V

        decorrelated_offset = U_perp @ Sigma_cat @ V_perp.T
        
        return decorrelated_offset.unsqueeze(0)
    
    @torch.no_grad()
    def generate(self, src_input, max_new_tokens, num_beams, decoder_start_token_id):
      
        
        if self._cached_updated_visual_prompts is None:
            # Visual side
            visual_offset = self._decorrelate_and_reconstruct(self.task_vector_vq, self.task_vector_vl)
            self._cached_updated_visual_prompts = self.visual_soft_prompts + visual_offset
            # Text side
            text_offset = self._decorrelate_and_reconstruct(self.task_vector_tq, self.task_vector_tl)
            base_text_prompts = self.text_prompt_mlp(self.text_soft_prompts)
            self._cached_updated_text_prompts = base_text_prompts + text_offset

        frames_feature = self.backbone(src_input['input_ids'].cuda(), src_input['src_length_batch'])
        attention_mask = src_input['attention_mask']
        inputs_embeds = self.sign_emb(frames_feature)
        inputs_embeds = self.embed_scale * inputs_embeds
        
        B = inputs_embeds.shape[0]
        updated_visual_prompts = self._cached_updated_visual_prompts.expand(B, -1, -1)

        encoder_inputs_embeds = torch.cat((updated_visual_prompts, inputs_embeds), dim=1)
        encoder_attention_mask = F.pad(attention_mask, (self.num_prompts, 0), value=1.)

        encoder = self.mbart.get_encoder()
        encoder_outputs = encoder(inputs_embeds=encoder_inputs_embeds, 
                                  attention_mask=encoder_attention_mask, 
                                  return_dict=True)

        updated_text_prompts = self._cached_updated_text_prompts.expand(B, -1, -1)
        start_token_ids = torch.full((B, 1), decoder_start_token_id, dtype=torch.long, device=encoder_inputs_embeds.device)
        start_token_embeds = self.mbart.model.decoder.embed_tokens(start_token_ids)
        initial_decoder_embeds = torch.cat([updated_text_prompts, start_token_embeds], dim=1)
        initial_decoder_mask = torch.ones_like(initial_decoder_embeds[..., 0])

        warm_up_outputs = self.mbart(
            encoder_outputs=encoder_outputs,
            decoder_inputs_embeds=initial_decoder_embeds,
            decoder_attention_mask=initial_decoder_mask,
            use_cache=True,
            return_dict=True
        )
        past_key_values = warm_up_outputs.past_key_values

        generated_tokens = self.mbart.generate(
            decoder_input_ids=start_token_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

        return generated_tokens
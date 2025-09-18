# *torch
from pickletools import optimize
# from sched import scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler as scheduler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from quantization.utils import progress_bar, inplace_quantize_layers, enable_calibrate, disable_calibrate

# *transformers
from transformers import MBartForConditionalGeneration, MBartTokenizer,MBartConfig
from transformers.models.mbart.modeling_mbart import shift_tokens_right

# *user-defined
# Renamed for clarity in the teacher-student context
from models import gloss_free_model_Prompt as StudentModel
from models import gloss_free_model as TeacherModel
from datasets import S2T_Dataset
import utils as utils

# *basic
import os
import time
import shutil
import argparse, json, datetime
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import yaml
import random
import test as test
import wandb
import copy
from pathlib import Path
from typing import Iterable, Optional
import math, sys
from loguru import logger

from hpman.m import _
import hpargparse

# *metric
from metrics import wer_list
from sacrebleu.metrics import BLEU, CHRF, TER
try:
    from nlgeval import compute_metrics
except:
    print('Please install nlgeval package.')


# *timm
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import NativeScaler

# global definition
from definition import *


class DistillationLoss(nn.Module):
    """
    This module computes the distillation loss.
    """
    def __init__(self, temp: float):
        super().__init__()
        self.temp = temp
    
    def forward(self, student_logits, teacher_logits):
        """
        Args:
            student_logits: Logits from the student model.
            teacher_logits: Logits from the teacher model.
        """
        # Soften probabilities with temperature
        soft_teacher_probs = F.softmax(teacher_logits / self.temp, dim=-1)
        soft_student_log_probs = F.log_softmax(student_logits / self.temp, dim=-1)

        # Compute KL divergence. The `batchmean` reduction averages the loss over the batch.
        distillation_loss = F.kl_div(
            soft_student_log_probs,
            soft_teacher_probs,
            reduction='batchmean'
        )
        
        # Scale the loss by T^2 as proposed in the original paper by Hinton et al.
        scaled_loss = (self.temp ** 2) * distillation_loss
        return scaled_loss


def get_args_parser():
    parser = argparse.ArgumentParser('Gloss-free Sign Language Translation script', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=80, type=int)

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint for the student model')
    
    # * Teacher-Student Distillation params
    parser.add_argument('--teacher-path', type=str, default='',
                        help='Path to the pretrained teacher model checkpoint.')
    parser.add_argument('--distillation-alpha', type=float, default=0.5,
                        help='Weight for the distillation loss (0 to 1).')
    parser.add_argument('--distillation-temp', type=float, default=2.0,
                        help='Temperature for softening probabilities in distillation.')

    # * Optimizer parameters
    # ... (rest of arguments are unchanged)
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1.0e-09)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='weight decay (default: 0.05)')

    # * Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1.0e-08, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    
     # * Baise params
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint for the student model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--config', type=str, default='./configs/config_gloss_free.yaml')

    # *Drop out params
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    # * Mixup params (Note: Mixup with distillation can be complex, use with caution)
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    
    # * data process params
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--resize', default=256, type=int)
    
    # * visualization
    parser.add_argument('--visualize', action='store_true')

    return parser


def load_custom_checkpoint(model: nn.Module, checkpoint_path: str, config: dict):
    """
    Loads a checkpoint with custom key re-mapping for the model.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint path not provided or does not exist: {checkpoint_path}")
        return

    print('***********************************')
    print(f'Loading custom parameters from: {checkpoint_path}')
    print('***********************************')
    
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    new_state_dict = OrderedDict()

    # The original logic expects weights under 'model' and 'text_decoder' keys
    model_weights = state_dict.get('model', {})
    text_decoder_weights = state_dict.get('text_decoder', {})

    for k, v in model_weights.items():
        if 'conv_2d' in k or 'conv_1d' in k:
            k = 'backbone.' + '.'.join(k.split('.')[2:])
            new_state_dict[k] = v
        if 'trans_encoder' in k:
            k = 'mbart.model.encoder.' + '.'.join(k.split('.')[2:])
            new_state_dict[k] = v

    for k, v in text_decoder_weights.items():
        if 'decoder' in k:
            k = 'mbart.model.decoder.' + '.'.join(k.split('.')[2:])
            new_state_dict[k] = v
        
    # *replace the word embedding from the base transformer
    transformer_path = os.path.join(config['model']['transformer'], 'pytorch_model.bin')
    if os.path.exists(transformer_path):
        model_dict = torch.load(transformer_path, map_location='cpu')
        for k, v in model_dict.items():
            if 'decoder.embed_tokens.weight' in k:
                k = 'mbart.' + k
                new_state_dict[k] = v
            if 'decoder.embed_positions.weight' in k:
                k = 'mbart.' + k
                new_state_dict[k] = v
    else:
        logger.warning(f"Base transformer for embeddings not found at: {transformer_path}")

    ret = model.load_state_dict(new_state_dict, strict=False)
    print('Missing keys: \n', '\n'.join(ret.missing_keys))
    print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))
    print('-----------------------------------')


def main(args, config):
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    print(f"Creating dataset:")
    tokenizer = MBartTokenizer.from_pretrained(config['model']['tokenizer'])
    # Data loaders remain the same
    train_data = S2T_Dataset(path=config['data']['train_label_path'], tokenizer = tokenizer, config=config, args=args, phase='train')
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=train_data.collate_fn, shuffle=True, pin_memory=args.pin_mem)
    dev_data = S2T_Dataset(path=config['data']['dev_label_path'], tokenizer = tokenizer, config=config, args=args, phase='val')
    dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dev_data.collate_fn, shuffle=False, pin_memory=args.pin_mem)
    test_data = S2T_Dataset(path=config['data']['test_label_path'], tokenizer = tokenizer, config=config, args=args, phase='test')
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=test_data.collate_fn, shuffle=False, pin_memory=args.pin_mem)
    
    print(f"Creating STUDENT model:")
    student_model = StudentModel(config, args)
    student_model = inplace_quantize_layers(student_model, len(train_dataloader) * 50, ptq = True if args.type == "PTQ" else False,
                              level = 'L', omse = True)
    student_model.to(device)
    
    # --- Load STUDENT model weights if fine-tuning ---
    if args.finetune:
        load_custom_checkpoint(student_model, args.finetune, config)

    n_parameters = utils.count_parameters_in_MB(student_model)
    print(f'Number of params in student: {n_parameters}M')

    # --- Create and prepare the TEACHER model ---
    
    teacher_model = TeacherModel(config, args)
    # Use the reusable function to load teacher weights
    load_custom_checkpoint(teacher_model, args.teacher_path, config)
    teacher_model.to(device)
    teacher_model.eval() # Set teacher to evaluation mode
    # Freeze teacher parameters
    for param in teacher_model.parameters():
        param.requires_grad = False
    print("Teacher model created and frozen.")
    print("\nFreezing all student model parameters by default...")
    for param in student_model.parameters():
        param.requires_grad = False

    params_to_train_names = []
    print("Unfreezing 'task_vector' and 'text_prompt_mlp' parameters...")
    for name, param in student_model.named_parameters():
        if 'task_vector' in name or 'text_prompt_mlp' in name:
            param.requires_grad = True
            params_to_train_names.append(name)
    
    print("\nParameters to be trained:")
    if not params_to_train_names:
        print("Warning: No parameters were unfrozen!")
    else:
        for name in params_to_train_names:
            print(name)
    
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params:,}")
    print("-" * 50)
    # Optimizer is created for the STUDENT model only
    optimizer = create_optimizer(args, student_model)
    lr_scheduler = scheduler.CosineAnnealingLR(optimizer=optimizer, eta_min=1e-8, T_max=args.epochs)
    loss_scaler = NativeScaler()

    # --- Define Loss Functions ---
    criterion_hard = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.2)
    criterion_soft = DistillationLoss(temp=args.distillation_temp)

    output_dir = Path(args.output_dir)
    if args.resume:
        print('Resuming STUDENT Model Parameters... ')
        checkpoint = torch.load(args.resume, map_location='cpu')
        student_model.load_state_dict(checkpoint['model'], strict=True)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        if not args.resume:
            logger.warning('Please specify the trained student model: --resume /path/to/best_checkpoint.pth')
        # Evaluation is done on the student model
        test_stats = evaluate(args, dev_dataloader, student_model, tokenizer, criterion_hard, config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BELU-4 of the student model on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f} ")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        student_model._cached_updated_visual_prompts = None
        student_model._cached_updated_text_prompts = None
        train_stats = train_one_epoch(
            args, student_model, teacher_model, criterion_hard, criterion_soft,
            train_dataloader, optimizer, device, epoch, config, loss_scaler
        )
        lr_scheduler.step(epoch)

        if args.output_dir:
            checkpoint_path = output_dir / 'checkpoint.pth'
            utils.save_on_master({
                'model': student_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, checkpoint_path)
        
        test_stats = evaluate(args, dev_dataloader, student_model, tokenizer, criterion_hard, config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BELU-4 of the student model on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f}")

        if max_accuracy < test_stats["belu4"]:
            max_accuracy = test_stats["belu4"]
            if args.output_dir:
                checkpoint_path = output_dir / 'best_checkpoint.pth'
                utils.save_on_master({
                    'model': student_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            
        print(f'Max BELU-4: {max_accuracy:.2f}%')
        wandb.log({
            'epoch': epoch + 1,
            'training/train_loss': train_stats['loss'],
            'training/loss_hard': train_stats['loss_hard'],
            'training/loss_soft': train_stats['loss_soft'],
            'dev/dev_loss': test_stats['loss'],
            'dev/Bleu_4': test_stats['belu4'],
            'dev/Best_Bleu_4': max_accuracy
        })

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(args, 
                    student_model: torch.nn.Module,
                    teacher_model: Optional[torch.nn.Module],
                    criterion_hard: nn.CrossEntropyLoss,
                    criterion_soft: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, config, loss_scaler):
    student_model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # Get student model output, which includes text and image logits
        student_txt_logits, student_img_logits = student_model(src_input, tgt_input)
        
        # --- Calculate Loss Components---
        label = tgt_input['input_ids'].reshape(-1)
        
        # 1. Hard loss component
        logits_for_hard_loss = student_txt_logits.reshape(-1, student_txt_logits.shape[-1])
        loss_hard = criterion_hard(logits_for_hard_loss, label.to(device, non_blocking=True))
        
        loss_soft_val = 0.0
        loss_soft = torch.tensor(0.0, device=device) # Initialize soft loss

        # 2. Soft loss components (if teacher exists)
        if teacher_model is not None:
            with torch.no_grad():
                teacher_txt_logits, teacher_img_logits = teacher_model(src_input, tgt_input)
            
            loss_soft_txt = criterion_soft(student_txt_logits, teacher_txt_logits)
            loss_soft_img = criterion_soft(student_img_logits, teacher_img_logits)
            loss_soft = (loss_soft_txt + loss_soft_img)/2
            loss_soft_val = loss_soft.item() # For logging purposes

        optimizer.zero_grad()


        loss_hard_scaled = loss_hard

        loss_hard_scaled.backward(retain_graph=True if teacher_model is not None else False)


        grads_from_hard = {
            name: param.grad.clone() for name, param in student_model.named_parameters() if param.requires_grad and param.grad is not None
        }

      
        optimizer.zero_grad()

        if teacher_model is not None:
            loss_soft.backward()

     
        with torch.no_grad():
            for name, param in student_model.named_parameters():
                if not param.requires_grad:
                    continue

                is_soft_group = 'task_vector_vq' in name or 'task_vector_tq' in name
                is_hard_group = 'task_vector_vl' in name or 'task_vector_tl' in name

                grad_s = param.grad
                grad_h = grads_from_hard.get(name)

                param.grad = None

                if is_soft_group:
                    if grad_s is not None:
                        param.grad = grad_s
                elif is_hard_group:

                    if grad_h is not None:
                        param.grad = grad_h
                else:
                    if grad_h is not None and grad_s is not None:
                        param.grad = grad_h + grad_s
                    elif grad_h is not None:

                        param.grad = grad_h
        
        
        # --- Step 6: Step the optimizer with the manually combined gradients ---
        optimizer.step()

        # The combined loss is now just for logging purposes
        loss_value = (loss_hard_scaled.item() + loss_soft_val) if teacher_model is not None else loss_hard.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_hard=loss_hard.item())
        metric_logger.update(loss_soft=loss_soft_val)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, dev_dataloader, model, tokenizer, criterion,  config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device):
    # The 'model' passed here is the student model. Teacher is not used for evaluation.
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []
 
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(dev_dataloader, 10, header)):
            out_logits = model(src_input, tgt_input)
            total_loss = 0.0
            label = tgt_input['input_ids'].reshape(-1)
            
            logits = out_logits.reshape(-1, out_logits.shape[-1])
            tgt_loss = criterion(logits, label.to(device))
            total_loss += tgt_loss
            metric_logger.update(loss=total_loss.item())
            
            output = model.generate(src_input, max_new_tokens=150, num_beams = 4,
                        decoder_start_token_id=tokenizer.lang_code_to_id['de_DE']
                        )

            tgt_input['input_ids'] = tgt_input['input_ids'].to(device)
            for i in range(len(output)):
                tgt_pres.append(output[i,:])
                tgt_refs.append(tgt_input['input_ids'][i,:])

    # --- Metric calculation remains the same ---
    if not tgt_pres or not tgt_refs:
        logger.warning("Evaluation produced no outputs to score.")
        return {'belu4': 0, 'loss': metric_logger.loss.global_avg}

    pad_tensor = torch.ones(200-len(tgt_pres[0])).to(device)
    tgt_pres[0] = torch.cat((tgt_pres[0],pad_tensor.long()),dim = 0)
    tgt_pres = pad_sequence(tgt_pres,batch_first=True,padding_value=PAD_IDX)

    pad_tensor = torch.ones(200-len(tgt_refs[0])).to(device)
    tgt_refs[0] = torch.cat((tgt_refs[0],pad_tensor.long()),dim = 0)
    tgt_refs = pad_sequence(tgt_refs,batch_first=True,padding_value=PAD_IDX)

    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)
    tgt_refs = tokenizer.batch_decode(tgt_refs, skip_special_tokens=True)

    bleu = BLEU()
    bleu_s = bleu.corpus_score(tgt_pres, [tgt_refs]).score
    metric_logger.meters['belu4'].update(bleu_s)

    print('* BELU-4 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.belu4, losses=metric_logger.loss))
    
    if args.eval:
        with open(args.output_dir+'/tmp_pres.txt','w') as f:
            for i in range(len(tgt_pres)):
                f.write(tgt_pres[i]+'\n')
        with open(args.output_dir+'/tmp_refs.txt','w') as f:
            for i in range(len(tgt_refs)):
                f.write(tgt_refs[i]+'\n')
        print('\n'+'*'*80)
        metrics_dict = compute_metrics(hypothesis=args.output_dir+'/tmp_pres.txt',
                           references=[args.output_dir+'/tmp_refs.txt'],no_skipthoughts=True,no_glove=True)
        print('*'*80)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Gloss-free Sign Language Translation script', parents=[get_args_parser()])
    _.parse_file(Path(__file__).resolve().parent)
    hpargparse.bind(parser, _)
    args = parser.parse_args()

    with open(args.config, 'r+',encoding='utf-8') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    
    os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
    if wandb.run:
        wandb.init(project='GF-SLT',config=config)
        wandb.run.name = args.output_dir.split('/')[-1]
        wandb.define_metric("epoch")
        wandb.define_metric("training/*", step_metric="epoch")
        wandb.define_metric("dev/*", step_metric="epoch")
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)
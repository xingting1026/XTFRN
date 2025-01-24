import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.losses import SupConLoss
from tensorboardX import SummaryWriter
from torch.nn import NLLLoss

def calculate_warmup_weight(current_epoch, start_epoch=30, end_epoch=60):
    """Calculate contrastive loss weight with warmup period"""
    if current_epoch < start_epoch:
        return torch.tensor(0.0).cuda()
        
    if current_epoch < end_epoch:
        progress = (current_epoch - start_epoch) / (end_epoch - start_epoch)
        return torch.tensor(0.1 * progress).cuda()  # 線性增加到0.1
    else:
        return torch.tensor(0.1).cuda()  # 固定在0.1

def default_train(train_loader, model, optimizer, writer, iter_counter, current_epoch, start_epoch=30, end_epoch=60):
    """
    Training function for BiRFN model with contrastive learning
    
    Args:
        train_loader: DataLoader for training
        model: BiRFN model
        optimizer: optimizer for training
        writer: tensorboard writer
        iter_counter: global iteration counter
        current_epoch: current training epoch
        start_epoch: epoch to start contrastive learning
        end_epoch: epoch to reach full contrastive learning weight
    """
    way = model.way
    shot = model.shots[0] 
    query_shot = model.shots[1]

    # 設定分類和對比損失
    target = torch.arange(way).repeat_interleave(query_shot).cuda()
    criterion_cls = NLLLoss().cuda()
    criterion_con = SupConLoss(temperature=0.1).cuda()  
    max_grad_norm = 10.0  
    
    # 記錄訓練指標
    writer.add_scalar('current_epoch', current_epoch, iter_counter)
    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('lr', lr, iter_counter)
    writer.add_scalar('scale', model.scale.item(), iter_counter)
    writer.add_scalar('W1', model.w1.item(), iter_counter)
    writer.add_scalar('W2', model.w2.item(), iter_counter)

    # 初始化平均指標
    avg_loss = 0
    avg_acc = 0
    avg_con_loss = 0
    avg_contra_weight = 0
    
    # 訓練迭代
    for i, (inp, _) in enumerate(train_loader):
        iter_counter += 1
        inp = inp.cuda()
        
        # 前向傳播
        log_prediction, contra_features = model(inp)
        
        # 計算分類損失
        cls_loss = criterion_cls(log_prediction, target)
        
        # 提取query樣本的特徵用於對比學習
        query_features = contra_features[way*shot:]
        query_labels = target
        
        # 計算對比損失
        con_loss = criterion_con(
            query_features.unsqueeze(1),  # [N, 1, D]
            query_labels,
        )
        
        # 計算帶warm up的權重
        contra_weight = calculate_warmup_weight(
            current_epoch,
            start_epoch,
            end_epoch
        )
        
        # 更新模型中的當前權重
        model.current_contra_weight = contra_weight.item()
        writer.add_scalar('contra_weight', contra_weight.item(), iter_counter)
        
        # 根據訓練階段決定是否使用對比損失
        if current_epoch >= start_epoch:
            loss = cls_loss + contra_weight * con_loss
            avg_con_loss += con_loss.item()
            avg_contra_weight += contra_weight.item()
        else:
            loss = cls_loss

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # 計算準確率
        loss_value = loss.item()
        _, max_index = torch.max(log_prediction, 1)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way

        # 更新平均指標
        avg_acc += acc
        avg_loss += loss_value

    # 計算epoch平均值
    avg_acc = avg_acc / (i + 1)
    avg_loss = avg_loss / (i + 1)
    avg_con_loss = avg_con_loss / (i + 1) if current_epoch >= start_epoch else 0
    avg_contra_weight = avg_contra_weight / (i + 1) if current_epoch >= start_epoch else 0

    # 記錄訓練指標
    writer.add_scalar('proto_loss', avg_loss, iter_counter)
    writer.add_scalar('train_acc', avg_acc, iter_counter)
    writer.add_scalar('contrastive_loss', avg_con_loss, iter_counter)
    writer.add_scalar('avg_contra_weight', avg_contra_weight, iter_counter)

    return iter_counter, avg_acc, avg_con_loss, avg_contra_weight
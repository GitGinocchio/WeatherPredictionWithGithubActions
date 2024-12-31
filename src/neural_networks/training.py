from pathlib import Path
from typing import Type
from tqdm import tqdm
import numpy as np
import sys

from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from utils.terminal import getlogger

logger = getlogger()

def validate(
        model: nn.Module, 
        val_dataloader: DataLoader, 
        device: torch.device, 
        loss_fn: nn.Module
        ) -> torch.Tensor:
    """
    performs cross-validation on model with validation dataloader
    returns loss on validation set

    keyword arguments:
    model      -- LSTM from /lstm/lstm.py
    val_loader -- validation dataloader
    device     -- torch device
    loss_fn    -- Mean Squared Error torch.nn.MSEloss
    """
    logger.debug("Validating model")
    logger.debug("Setting model to evaluation mode")
    model.eval()
    val_loss = 0
    logger.debug("Starting evaluation")
    for batch_in, batch_out in val_dataloader:
        pred = model(batch_in.to(device))
        loss = loss_fn(pred, batch_out.to(device))
        val_loss += loss
    val_loss /= len(val_dataloader)
    logger.debug("Validation complete. Loss: {val_loss:.4f}")
    return val_loss

def train(
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader, 
        model: nn.Module, 
        device: torch.device, 
        loss_fn: nn.Module, 
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler, 
        num_epochs: int, 
        model_name: str, 
        patience: int,
        disable_patience : bool,
        save_dir: str
        ) -> tuple[list, list]:
    """
    trains LSTM on train dataset
    ---

    returns list containing train and val loss per epoch

    keyword arguments:

    train_loader    -- DataLoader for train dataset

    val_loader      -- DataLoader for validation dataset

    model           -- LSTM model

    device          -- torch device 

    loss_fn         -- Mean Squared Error torch.nn.MSEloss

    optimizer       -- torch Adam

    num_epochs      -- number of training iterations through the train dataset

    model_save_name -- name of the model to save, include .pth

    patience        -- number of epochs past validation loss improving to continue training

    result_dir      -- Path to model and result file save location 
    """
    model.to(device)
    train_loss_list = []
    val_loss_list = []
    best_val_loss = np.inf
    patience_counter = 0
    for epoch_idx in range(num_epochs):
        model.train()
        if patience_counter >= patience:
            break
        else:
            tr_epoch_loss = 0
            for batch_in, batch_out in train_dataloader:
                pred = model(batch_in.to(device))
                train_loss = loss_fn(pred, batch_out.to(device))
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                tr_epoch_loss += train_loss
            tr_epoch_loss /= len(train_dataloader)
            train_loss_list.append((epoch_idx + 1, float(tr_epoch_loss.cpu().detach())))
            
            val_loss = validate(model=model, val_dataloader=val_dataloader, device=device, loss_fn=loss_fn)
            val_loss_list.append((epoch_idx+1, float(val_loss.cpu().detach())))

            logger.info(f"Epoch: {epoch_idx+1:>{len(str(num_epochs))}}/{num_epochs} (Train loss: {tr_epoch_loss:.5f}, Validation loss: {val_loss:.5f})")

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(),Path(save_dir).joinpath(model_name))
            elif not disable_patience:
                patience_counter += 1

    return train_loss_list, val_loss_list

def test(
        model: nn.Module, 
        model_name: str, 
        test_dataloader: DataLoader, 
        device: torch.device, 
        loss_fn: nn.Module, 
        save_dir: Path
        ) -> float:
    """
    evaluates model performance on test dataset

    keyword arguments:
    model           -- LSTM from /lstm/lstm.py
    model_save_name -- name to reload best saved model, including .pth.tar
    test_loader     -- DataLoader for test dataset
    device          -- torch device
    loss_fn         -- Mean Squared Error torch.nn.MSEloss
    save_dir        -- Path to save test result file to
    """
    checkpoint = torch.load(Path(save_dir).joinpath(model_name), weights_only=True)
    model.load_state_dict(checkpoint)

    test_loss = 0
    for batch_in, batch_out in test_dataloader:
        pred = model(batch_in.to(device))
        loss = loss_fn(pred, batch_out.to(device))
        test_loss += loss
    test_loss /= len(test_dataloader)  
    logger.info(f"Test loss: {test_loss:.5f}")
    return float(test_loss.cpu().detach())
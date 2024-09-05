from typing import List
import os

import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Select GPU 0
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def train_nnet(nnet: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               epochs: int, lr: float, weight_decay: float, model_dir: str,
               checkpoint_interval: int, cube_len: int, patience: int = 5, device=None):
    # Set device if not provided
    if device is None:
        device = get_device()

    # Move model to the device
    nnet.to(device)

    # Initialize optimizer, criterion, and scheduler
    optimizer = optim.Adam(nnet.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_value = nn.MSELoss()  # nn.SmoothL1Loss() Loss for value network
    criterion_policy = nn.CrossEntropyLoss()  # Loss for policy network
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

    # Early stopping variables
    best_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    # Create a separate folder for checkpoints
    cube_dir = os.path.join(model_dir, f"{cube_len}x{cube_len}")
    checkpoint_dir = os.path.join(cube_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Lists to store loss history
    train_policy_losses, train_value_losses = [], []
    val_policy_losses, val_value_losses = [], []
    progress_bar = tqdm(total=epochs, desc="Training Progress")
    # Training loop
    for epoch in range(epochs):
        nnet.train()
        total_policy_loss, total_value_loss = 0.0, 0.0

        # Train for one epoch with progress bar
        for states_batch, policy_targets_batch, value_targets_batch in train_loader:
            states_batch = states_batch.to(device)
            policy_targets_batch = policy_targets_batch.to(device)
            value_targets_batch = value_targets_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            policy_output, value_output = nnet(states_batch)

            # Loss calculation
            policy_loss = criterion_policy(policy_output, policy_targets_batch)
            value_loss = criterion_value(value_output.view(-1), value_targets_batch)

            loss = policy_loss + value_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        # Validation phase
        val_policy_loss, val_value_loss = validate_nnet(nnet, val_loader, criterion_policy,
                                                        criterion_value, device)

        # Track losses
        train_policy_losses.append(total_policy_loss / len(train_loader))
        train_value_losses.append(total_value_loss / len(train_loader))
        val_policy_losses.append(val_policy_loss)
        val_value_losses.append(val_value_loss)

        # Update progress bar after each epoch
        progress_bar.update(1)
        # Print summary for this epoch (after the batch loop)
        print(f'Epoch [{epoch+1}/{epochs}], Train Policy Loss: {total_policy_loss / len(train_loader):.3f}, '
              f'Train Value Loss: {total_value_loss / len(train_loader):.3f}, '
              f'Val Policy Loss: {val_policy_loss:.3f}, Val Value Loss: {val_value_loss:.3f}')

        # Early stopping logic and best model tracking
        current_loss = val_policy_loss + val_value_loss
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch + 1
            best_model_state = {
                'epoch': best_epoch,
                'model_state_dict': nnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_loss,
                'train_policy_losses': train_policy_losses,
                'train_value_losses': train_value_losses,
                'val_policy_losses': val_policy_losses,
                'val_value_losses': val_value_losses
            }
            patience_counter = 0
        else:
            patience_counter += 1

        # Save checkpoints at the specified interval
        if (epoch + 1) % checkpoint_interval == 0:
            save_model(nnet, optimizer, epoch + 1, current_loss, checkpoint_dir, train_policy_losses,
                       train_value_losses, val_policy_losses, val_value_losses)

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}. Best model found at epoch {best_epoch}.")
            break

        # Step the learning rate scheduler
        scheduler.step(current_loss)

    # Save the best model at the end of training
    if best_model_state:
        save_best_model(best_model_state, cube_dir)

    return best_loss


def save_model(nnet: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, val_loss: float,
               model_dir: str, train_policy_losses: List[float], train_value_losses: List[float],
               val_policy_losses: List[float], val_value_losses: List[float]):
    """
    Saves the model state, optimizer state, epoch, validation loss, and loss history to a checkpoint file.

    Args:
        nnet (nn.Module): The neural network model to save.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        epoch (int): The current epoch.
        val_loss (float): The validation loss to save.
        model_dir (str): The directory to save the model in.
        train_policy_losses (List[float]): List of training policy losses.
        train_value_losses (List[float]): List of training value losses.
        val_policy_losses (List[float]): List of validation policy losses.
        val_value_losses (List[float]): List of validation value losses.
    """
    save_path = os.path.join(model_dir, f"nnet_checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': nnet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'train_policy_losses': train_policy_losses,
        'train_value_losses': train_value_losses,
        'val_policy_losses': val_policy_losses,
        'val_value_losses': val_value_losses
    }, save_path)
    print(f"Checkpoint saved at {save_path}")


def save_best_model(best_model_state: dict, model_dir: str):
    """
    Saves the best model at the end of training.

    Args:
        best_model_state (dict): The state dictionary of the best model.
        model_dir (str): The directory to save the best model.
    """
    save_path = os.path.join(model_dir, f"best_model_epoch_{best_model_state['epoch']}.pt")
    torch.save(best_model_state, save_path)
    print(f"Best model saved at {save_path}")


def validate_nnet(nnet: torch.nn.Module, val_loader: DataLoader,
                  policy_loss_fn: torch.nn.CrossEntropyLoss,
                  value_loss_fn: torch.nn.SmoothL1Loss, device):
    """
    Validate the neural network on the validation set with separate losses.
    """
    nnet.eval()
    val_policy_loss = 0.0
    val_value_loss = 0.0
    with torch.no_grad():
        for inputs, policy_targets, value_targets in val_loader:
            inputs = inputs.to(device)
            policy_targets, value_targets = policy_targets.to(device), value_targets.to(device)
            policy_outputs, value_outputs = nnet(inputs)

            policy_loss = policy_loss_fn(policy_outputs, policy_targets)
            value_loss = value_loss_fn(value_outputs.view(-1), value_targets)

            val_policy_loss += policy_loss.item()
            val_value_loss += value_loss.item()

    return val_policy_loss / len(val_loader), val_value_loss / len(val_loader)


def load_model(model_file: str, nnet: nn.Module, optimizer: torch.optim.Optimizer = None,
               device=None, eval_mode: bool = True):
    """
    Loads the model, optimizer, and loss history from the checkpoint file and optionally sets the model
    to evaluation mode.

    Args:
        model_file (str): The path to the checkpoint file.
        nnet (nn.Module): The model to load the state dict into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state dict into.
        device (torch.device, optional): The device to load the model onto.
        eval_mode (bool): Whether to set the model in evaluation mode (default is True).

    Returns:
        Tuple: The epoch, validation loss, and loss history (train and validation losses).
    """
    if device is None:
        checkpoint = torch.load(model_file)
    else:
        checkpoint = torch.load(model_file, map_location=device)

    # Load model state
    nnet.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state if provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']

    # Load loss history
    train_policy_losses = checkpoint['train_policy_losses']
    train_value_losses = checkpoint['train_value_losses']
    val_policy_losses = checkpoint['val_policy_losses']
    val_value_losses = checkpoint['val_value_losses']

    print(f"Model and optimizer loaded from {model_file} (Epoch {epoch})")

    # Set the model to evaluation mode if specified
    if eval_mode:
        nnet.eval()

    return epoch, val_loss, train_policy_losses, train_value_losses, val_policy_losses, val_value_losses


def plot_losses(train_policy_losses: List[float], train_value_losses: List[float],
                val_policy_losses: List[float], val_value_losses: List[float], model_dir: str):
    """
    Plots the training and validation policy and value losses.

    Args:
        train_policy_losses (List[float]): List of training policy losses.
        train_value_losses (List[float]): List of training value losses.
        val_policy_losses (List[float]): List of validation policy losses.
        val_value_losses (List[float]): List of validation value losses.
        model_dir (str): The directory to save the plots.
    """
    # Plot policy losses
    plt.figure()
    plt.plot(train_policy_losses, label='Train Policy Loss')
    plt.plot(val_policy_losses, label='Val Policy Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Policy Loss')
    plt.legend()
    plt.title('Training and Validation Policy Loss')
    plt.savefig(os.path.join(model_dir, 'policy_loss_plot.png'))
    plt.show()

    # Plot value losses
    plt.figure()
    plt.plot(train_value_losses, label='Train Value Loss')
    plt.plot(val_value_losses, label='Val Value Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Value Loss')
    plt.legend()
    plt.title('Training and Validation Value Loss')
    plt.savefig(os.path.join(model_dir, 'value_loss_plot.png'))
    plt.show()

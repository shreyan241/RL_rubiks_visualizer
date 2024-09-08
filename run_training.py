import argparse
import os
import random
from typing import List
import torch
from torch.utils.data import DataLoader, ConcatDataset
from Models import ResnetModel
from utils import load_dataset, train_nnet, get_device


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Rubik's Cube Solver Training")
    parser.add_argument('--cube_len', type=int, required=True, help="Length of the cube's side (e.g., 3 for a 3x3x3 cube)")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing training data")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory to save trained models")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 regularization weight decay")
    parser.add_argument('--checkpoint_interval', type=int, default=10, help="Interval between saving checkpoints")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--validation_size', type=float, required=True,
                        help="Proportion of files for validation (e.g., 0.2 for 20%)")
    parser.add_argument('--total_data_files', type=int, required=True, help="Total number of dataset files to use")
    parser.add_argument('--maxback', type=int, required=True, help="Maximum number of back moves for data generation")
    parser.add_argument('--resume_checkpoint', type=str, help="Path to the checkpoint file to resume training from", default=None)
    args = parser.parse_args()

    # Set device to GPU if available, otherwise fall back to CPU
    device = get_device()
    print(f"Using device: {device}")

    # Ensure model directory exists
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Get all available file indices based on total_data_files
    file_indices = list(range(args.total_data_files))

    # Calculate the number of validation files
    num_val_files = max(1, int(len(file_indices) * args.validation_size))

    # Randomly select validation files
    val_indices = random.sample(file_indices, num_val_files)

    # Remaining files are for training
    train_indices = [i for i in file_indices if i not in val_indices]

    if len(train_indices) == 0:
        raise ValueError("No training files selected. Adjust 'total_data_files' or 'validation_size'.")

    # Load datasets for training and validation
    print(f"Loading {len(train_indices)} training files and {len(val_indices)} validation files.")
    train_data = load_files(args.data_dir, train_indices, args.cube_len, args.maxback)
    val_data = load_files(args.data_dir, val_indices, args.cube_len, args.maxback)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    # Initialize the model based on cube size
    state_dim = 6 * args.cube_len**2
    nnet = ResnetModel(state_dim=state_dim, one_hot_depth=6, h1_dim=5000, resnet_dim=1000, num_resnet_blocks=4,
                       policy_out_dim=12, value_out_dim=1, batch_norm=True, dropout=0.1)

    # Move the model to the GPU (if available)
    nnet = nnet.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(nnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Load from checkpoint if provided
    start_epoch = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=True)
        nnet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from checkpoint: {args.resume_checkpoint}, starting at epoch {start_epoch}")

    # Train the model
    print(f"Starting training with {len(train_loader)} batches.")
    train_nnet(nnet=nnet, train_loader=train_loader, val_loader=val_loader,
               epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
               model_dir=args.model_dir, checkpoint_interval=args.checkpoint_interval,
               patience=args.patience, cube_len=args.cube_len, device=device,
               start_epoch=start_epoch, optimizer=optimizer)


def load_files(data_dir: str, indices: List[int], cube_len, maxback: int) -> ConcatDataset:
    """
    Load multiple dataset files based on their indices.

    Args:
        data_dir (str): The directory where the dataset files are stored.
        indices (List[int]): The list of file indices to load.
        maxback (int): Maximum number of back moves (for loading files with maxback as part of the filename).

    Returns:
        ConcatDataset: The concatenated dataset from all selected files.
    """
    datasets = []
    cube_dir = os.path.join(data_dir, f"{cube_len}x{cube_len}")
    for idx in indices:
        file_path = os.path.join(cube_dir, f"maxback{maxback}_idx{idx}.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file {file_path} not found.")
        dataset = load_dataset(file_path)
        datasets.append(dataset)

    return ConcatDataset(datasets)


if __name__ == "__main__":
    main()
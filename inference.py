import matplotlib.pyplot as plt
from argparse import ArgumentParser
import torch
import os

def plot_loss_accuracy(train_loss, train_accuracy, val_loss, val_accuracy):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    #removing the first epoch since loss was a very high initial estimate
    plt.plot(epochs[1:], train_loss[1:], label='Train Loss', color='blue', linestyle='-')
    plt.plot(epochs[1:], val_loss[1:], label='Validation Loss', color='red', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs Epochs')

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='Train Accuracy', color='green', linestyle='-')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy', color='orange', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs Epochs')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path',default='from_scratch_ckpt/checkpoint_10.pth', 
                        type=str, help='Path to checkpoint.pth file')
    args = parser.parse_args()

    #get absolute path to checkpoint
    ckpt_path = os.path.abspath(args.checkpoint_path)
    checkpoint = torch.load(ckpt_path)

    training_losses = checkpoint['training_losses']
    validation_losses = checkpoint['validation_losses']
    training_accs = checkpoint['training_accs']
    validation_accs = checkpoint['validation_accs']


    plot_loss_accuracy(training_losses, training_accs, validation_losses, validation_accs)
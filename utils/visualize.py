# Visualization utilities for MNIST data

import matplotlib.pyplot as plt
import numpy as np


def visualize_samples(images, labels, num_samples=10, title="MNIST Samples", save_path=None):
    """Visualize sample MNIST images.
    
    Args:
        images: Array of shape (N, 28, 28) with pixel values [0,1]
        labels: Array of shape (N,) with digit labels 0-9
        num_samples: Number of samples to display
        title: Title for the plot
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f'Label: {labels[i]}')
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        try:
            plt.show()
        except:
            # Fallback: save to file if display is not available
            fallback_path = f"{title.lower().replace(' ', '_')}.png"
            plt.savefig(fallback_path)
            print(f"Display not available. Plot saved to {fallback_path}")
    
    plt.close(fig)


def plot_training_history(train_losses, val_losses=None, train_acc=None, val_acc=None):
    """Plot training history.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        train_acc: List of training accuracies (optional)
        val_acc: List of validation accuracies (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss')
    if val_losses:
        ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    if train_acc and val_acc:
        ax2.plot(train_acc, label='Training Accuracy')
        ax2.plot(val_acc, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Visualization utilities loaded.")
    print("Use visualize_samples(images, labels) to view MNIST data")
    print("Use plot_training_history(losses) to plot training curves")

import matplotlib.pyplot as plt

def plot_performance(loss_history, accuracy_history):
    """
    Plots the Loss and Accuracy curves side by side.
    """
    # Create a figure with 2 subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Graph 1: Loss vs Epochs
    ax1.plot(loss_history, label='Training Loss', color='red')
    ax1.set_title('Loss vs Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE Loss')
    ax1.grid(True)
    ax1.legend()

    # Graph 2: Accuracy vs Epochs
    # Handle case where accuracy might be empty (e.g., if used for regression later)
    if accuracy_history:
        ax2.plot(accuracy_history, label='Training Accuracy', color='blue')
        ax2.set_title('Accuracy vs Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        ax2.legend()

    # Display the plots
    plt.tight_layout()
    plt.show()
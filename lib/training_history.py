import matplotlib.pyplot as plt

def smooth_curve(points, factor=0.9):
    """
    Smoothing function using Exponential Moving Average.
    factor: 0.0 (no smoothing) to 1.0 (max smoothing). 0.9 is standard.
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_performance(loss_history, accuracy_history):
    """
    Plots the Loss and Accuracy curves side by side.
    """
    # Create a figure with 2 subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Graph 1: Loss vs Epochs
    # Loss naturally looks smooth usually, so we plot it as is.
    ax1.plot(loss_history, label='Training Loss', color='red')
    ax1.set_title('Loss vs Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE Loss')
    ax1.grid(True)
    ax1.legend()

    # Graph 2: Accuracy vs Epochs
    if accuracy_history:
        # --- NEW: Apply smoothing before plotting ---
        smooth_acc = smooth_curve(accuracy_history, factor=0.85)
        
        # Plot the smoothed line
        ax2.plot(smooth_acc, label='Training Accuracy', color='blue', linewidth=2)
        
        # Optional: Plot the original faint line behind it to show the real data 
        # (Remove the next line if you want ONLY the smooth line)
        ax2.plot(accuracy_history, color='blue', alpha=0.2) 
        
        ax2.set_title('Accuracy vs Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        ax2.legend()

    # Display the plots
    plt.tight_layout()
    plt.show()
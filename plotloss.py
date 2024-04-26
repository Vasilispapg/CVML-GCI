# plot the loss
import matplotlib.pyplot as plt
import numpy as np
def plot_loss(file_path='loss.txt'):
    with open(file_path, 'r') as f:
        data = f.readlines()
    
    all_losses = []
    start_index = 0
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))  # Generate colors for each epoch

    plt.figure(figsize=(20, 5))
    for idx, line in enumerate(data):
        losses = line.strip().strip('[]').split(',')
        epoch_losses = [float(loss.strip()) for loss in losses if loss.strip()]
        end_index = start_index + len(epoch_losses)
        
        # Plot each epoch with different colors
        plt.plot(range(start_index, end_index), epoch_losses, marker='x', linestyle='-', color=colors[idx], label=f'Epoch {idx + 1}')
        
        # Optionally add a vertical line to denote the end of an epoch
        plt.axvline(x=end_index - 1, color='grey', linestyle='--', alpha=0.5)
        
        start_index = end_index
        all_losses.extend(epoch_losses)

    plt.title('Loss vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
    
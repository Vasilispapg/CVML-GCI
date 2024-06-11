import matplotlib.pyplot as plt
import numpy as np

def plot_loss(type=1, file_path='loss.txt', eval_file_path='evaluation_loss.txt'):
    # Read training loss data

    with open(file_path, 'r') as f:
        data = f.readlines()
    
    # Read evaluation loss data
    with open(eval_file_path, 'r') as f:
        dataEval = f.readlines()
    
    # Extract and parse losses
    all_losses = []
    eval_losses = []
    start_index = 0
    eval_start_index = 0
    
    # Generate colors for each epoch
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    colorseval = plt.cm.viridis(np.linspace(0, 1, len(dataEval)))
    
    if type == 1:
        plt.figure(figsize=(20, 5))
        
        # Plot training loss
        for idx, line in enumerate(data):
            losses = line.strip().strip('[]').split(',')
            epoch_losses = [float(loss.strip()) for loss in losses if loss.strip()]
            end_index = start_index + len(epoch_losses)
            
            plt.plot(range(start_index, end_index), epoch_losses, marker='x', linestyle='-', color=colors[idx], label=f'Training Epoch {idx + 1}')
            plt.axvline(x=end_index - 1, color='grey', linestyle='--', alpha=0.5)
            
            start_index = end_index
            all_losses.extend(epoch_losses)
        
        # Plot evaluation loss
        for idx, line in enumerate(dataEval):
            losses = line.strip().strip('[]').split(',')
            epoch_losses = [float(loss.strip()) for loss in losses if loss.strip()]
            eval_end_index = eval_start_index + len(epoch_losses)
            
            plt.plot(range(eval_start_index, eval_end_index), epoch_losses, marker='o', linestyle='-', color=colorseval[idx], label=f'Evaluation Epoch {idx + 1}', alpha=0.6)
            plt.axvline(x=eval_end_index - 1, color='grey', linestyle='--', alpha=0.5)
            
            eval_start_index = eval_end_index
            eval_losses.extend(epoch_losses)
        
        plt.title('Loss vs. Iteration (Training and Evaluation)')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    elif type == 2:
        fig, axs = plt.subplots(2, 1, figsize=(20, 10))
        
        # Plot training loss
        for idx, line in enumerate(data):
            losses = line.strip().strip('[]').split(',')
            epoch_losses = [float(loss.strip()) for loss in losses if loss.strip()]
            end_index = start_index + len(epoch_losses)
            
            axs[0].plot(range(start_index, end_index), epoch_losses, marker='x', linestyle='-', color=colors[idx], label=f'Epoch {idx + 1}')
            axs[0].axvline(x=end_index - 1, color='grey', linestyle='--', alpha=0.5)
            
            start_index = end_index
            all_losses.extend(epoch_losses)
        
        axs[0].set_title('Training Loss vs. Iteration')
        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('Loss')
        axs[0].grid(True)
        axs[0].legend()
        
        # Plot evaluation loss
        for idx, line in enumerate(dataEval):
            losses = line.strip().strip('[]').split(',')
            epoch_losses = [float(loss.strip()) for loss in losses if loss.strip()]
            eval_end_index = eval_start_index + len(epoch_losses)
            
            axs[1].plot(range(eval_start_index, eval_end_index), epoch_losses, marker='o', linestyle='-', color=colorseval[idx], label=f'Epoch {idx + 1}')
            axs[1].axvline(x=eval_end_index - 1, color='grey', linestyle='--', alpha=0.5)
            
            eval_start_index = eval_end_index
            eval_losses.extend(epoch_losses)
        
        axs[1].set_title('Evaluation Loss vs. Iteration')
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Loss')
        axs[1].grid(True)
        axs[1].legend()
        
        plt.tight_layout()
        plt.show()


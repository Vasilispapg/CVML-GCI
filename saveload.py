import torch


def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer=None):
    print("=> Loading checkpoint")

    # Handle case where model might be wrapped inside a tuple
    if isinstance(model, tuple):
        print("Model is a tuple, extracting the model instance.")
        model = model[0]  # Assuming the model is the first element in the tuple

    model_state = checkpoint['model']
    current_model_state = model.state_dict()

    # Filter out unnecessary keys
    filtered_model_state = {
        k: v for k, v in model_state.items()
        if k in current_model_state and current_model_state[k].size() == v.size()
    }

    model.load_state_dict(filtered_model_state, strict=False)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer

    return model


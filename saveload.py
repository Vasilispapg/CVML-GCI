import torch


def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if(optimizer is None):
        return model
    optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer


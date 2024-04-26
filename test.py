import tarfile

def inspect_checkpoint(file_path):
    try:
        # Trying different modes might help identify the correct format
        with tarfile.open(file_path, "r:gz") as tar:  # for GZip compressed files
            tar.list()
    except tarfile.ReadError as e:
        print(f"Failed to open as GZip: {e}")
        try:
            with tarfile.open(file_path, "r") as tar:  # for uncompressed tar files
                tar.list()
        except tarfile.ReadError as e:
            print(f"Failed to open as plain tar: {e}")

inspect_checkpoint(r'model_checkpoint.pth.tar')

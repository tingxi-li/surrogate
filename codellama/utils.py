import os 
import json
import torch


def find_folders_with_checkpoints(ckpt_dir):
    folders = []
    for root, dirs, files in os.walk(ckpt_dir):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            if any(name.startswith("checkpoint-") for name in os.listdir(folder_path)):
                folders.append(folder_path)
    return folders


def get_latest_checkpoint(folder):
    checkpoint_dirs = [d for d in os.listdir(folder) if d.startswith("checkpoint-")]
    if not checkpoint_dirs:
        print(f"[WARNING] No checkpoint folders found in {folder}")
        return None

    # Sort based on the step number
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]), reverse=True)
    latest_checkpoint = checkpoint_dirs[0]
    return os.path.join(folder, latest_checkpoint)


def load_model_from_checkpoint(folder):
    config_path = os.path.join(folder, "config.json")
    if not os.path.exists(config_path):
        print(f"[WARNING] config.json not found in {folder}")
        return None, None

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            print(f"[INFO] Loaded config from {config_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load config.json: {e}")
        return None, None

    # Get the latest checkpoint folder
    latest_ckpt_dir = get_latest_checkpoint(folder)
    if not latest_ckpt_dir:
        print(f"[WARNING] No valid checkpoint found in {folder}")
        return None, None

    print(f"[INFO] Latest checkpoint found: {latest_ckpt_dir}")
    return latest_ckpt_dir, config

if __name__ == "__main__":
    import pdb
    ckpt_dir = "/opt/dlami/nvme/surrogate_ckpt"
    folders = find_folders_with_checkpoints(ckpt_dir)
    for folder in folders:
        latest_checkpoint, config = load_model_from_checkpoint(folder)
        if latest_checkpoint and config:
            print(f"Latest checkpoint: {latest_checkpoint}")
            print(f"Config: {config}")
            pdb.set_trace()
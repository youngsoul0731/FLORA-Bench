import os
from datasets import load_dataset
from pathlib import Path

# --- 1. Configure for Mirror Site Download ---
# Set the environment variable to use the mirror site.
# This must be done BEFORE importing or using any huggingface_hub functions.
mirror_site = "https://hf-mirror.com"
os.environ['HF_ENDPOINT'] = mirror_site
print(f"✅ Using mirror site for downloads: {os.environ.get('HF_ENDPOINT')}")


# --- 2. Define dataset information and local save path ---
# Dataset name on the Hugging Face Hub
dataset_name = "YuanshuoZhang/FLORA-Bench"
# Define the name of the local directory to save the dataset
save_directory = "datasets_checkpoints"

# Get the current working directory
current_working_directory = Path.cwd()
# Construct the full save path
save_path = current_working_directory / save_directory


# --- 3. Create the target directory ---
# If the directory does not exist, create it
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Directory created: '{save_path}'")
else:
    print(f"Directory already exists: '{save_path}'")


# --- 4. Download and save the dataset ---
try:
    print(f"Starting download of dataset '{dataset_name}' from the mirror...")
    
    # Load the dataset. The library will automatically use the HF_ENDPOINT.
    dataset = load_dataset(dataset_name)
    
    # Save all splits of the dataset (e.g., 'train', 'test') to disk
    # The dataset will be saved in the efficient Arrow format at the specified path.
    dataset.save_to_disk(save_path)
    
    print("\n--------------------------------------------------")
    print("🎉 Dataset downloaded and saved successfully!")
    print(f"Dataset '{dataset_name}' has been saved to the following path:")
    print(str(save_path))
    print("--------------------------------------------------")

except Exception as e:
    print(f"\nAn error occurred during download or saving: {e}")
    print("\nPlease check the following:")
    print("1. Your internet connection is stable.")
    print(f"2. The mirror '{mirror_site}' is accessible.")
    print(f"3. The dataset name '{dataset_name}' is correct.")
    print("4. You have sufficient disk space.")
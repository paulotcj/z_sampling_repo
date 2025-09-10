import os
import shutil
from datasets import load_dataset, get_dataset_config_names, get_dataset_infos

#-------------------------------------------------------------------------
def clean_datasets_cache():
    """Cleans the Hugging Face datasets cache directory."""
    cache_dir = os.path.expanduser(os.path.join("~", ".cache", "huggingface"))
    print(f'cache_dir: {cache_dir}')
    exit()
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print(f"Datasets cache directory '{cache_dir}' cleaned successfully.")
        except Exception as e:
            print(f"Error cleaning datasets cache: {e}")
    else:
        print(f"Datasets cache directory '{cache_dir}' does not exist.")
#-------------------------------------------------------------------------


clean_datasets_cache()
#pip install -U "huggingface_hub[cli]"
from huggingface_hub import snapshot_download

# Downloads the entire model repo to your local folder
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir="./all-MiniLM-L6-v2",
    local_dir_use_symlinks=False
)
print("Download complete!")

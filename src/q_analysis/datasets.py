import gdown
import tarfile
from pathlib import Path
import os
from collections import defaultdict
from typing import Dict, List

def read_coauthors_dataset(data_dir: str) -> Dict[int, List[List[int]]]:
    dataset_path = Path(data_dir)
    simplices_by_year = defaultdict(list)

    with open(dataset_path / "coauth-DBLP-nverts.txt", 'r') as f_nverts:
        num_vertices_per_simplex = [int(line.strip()) for line in f_nverts]

    with open(dataset_path / "coauth-DBLP-simplices.txt", 'r') as f_vertices:
        concatenated_vertices = [int(line.strip()) for line in f_vertices]

    with open(dataset_path / "coauth-DBLP-times.txt", 'r') as f_times:
        simplex_timestamps = [int(line.strip()) for line in f_times]

    current_vertex_read_index = 0

    for simplex_size, year in zip(num_vertices_per_simplex, simplex_timestamps):
        start_index = current_vertex_read_index
        end_index = current_vertex_read_index + simplex_size
        current_simplex_vertices = concatenated_vertices[start_index:end_index]
        simplices_by_year[year].append(current_simplex_vertices)
        current_vertex_read_index = end_index

    return dict(sorted(simplices_by_year.items()))

DATASETS = {
    "coauthors": {
        "id": "15YpIK8vvzQJXyQC4bt-Sz951e4eb0rc_",
        "dir": "coauth-DBLP",
        "loader": read_coauthors_dataset
    }
}

def load_dataset(name: str, cache_dir: str = None, verbose: bool = True) -> Path:
    """
    Loads a dataset, downloading it if necessary.

    Args:
        name (str): The name of the dataset to load.
        cache_dir (str, optional): The directory to cache the dataset in. 
                                   Defaults to ~/.q_analysis_datasets.
        verbose (bool, optional): Whether to print download progress. Defaults to True.

    Returns:
        Path: The path to the dataset directory.
    """
    if name not in DATASETS:
        raise ValueError(f"Dataset '{name}' not recognized. Available datasets: {list(DATASETS.keys())}")

    if cache_dir is None:
        cache_dir = Path.home() / ".q_analysis_datasets"
    else:
        cache_dir = Path(cache_dir)

    dataset_info = DATASETS[name]
    dataset_id = dataset_info["id"]
    dataset_dir_name = dataset_info["dir"]
    dataset_loader = dataset_info["loader"]
    dataset_path = cache_dir / dataset_dir_name
    archive_path = cache_dir / f"{name}.tar.gz"

    if not dataset_path.exists():
        os.makedirs(cache_dir, exist_ok=True)
        
        if verbose:
            print(f"Downloading {name} dataset...")
        gdown.download(id=dataset_id, output=str(archive_path), quiet=not verbose)
        
        if verbose:
            print(f"Extracting {name} dataset...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=cache_dir)
        
        os.remove(archive_path)
    
    dataset = dataset_loader(dataset_path)
    return dataset 


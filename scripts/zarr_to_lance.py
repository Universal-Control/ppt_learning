import lance
import pyarrow as pa
import numpy as np
import ray
from pathlib import Path
import zarr
import tyro
import json
import pickle
import tqdm
import time

def zarr_apply(zarr_obj, func, *args, **kwargs):
    """
    Get the shape structure of zarr object, returns nested dictionary
    
    Args:
        zarr_obj: zarr.Group or zarr.Array object
        
    Returns:
        dict: Nested dictionary, final values are shape tuples
    """
    if isinstance(zarr_obj, zarr.Array):
        # If it's an array, directly return shape
        return func(zarr_obj, *args, **kwargs)
    elif isinstance(zarr_obj, zarr.Group):
        # If it's a group, recursively process all sub-items
        shape_dict = {}
        for key in zarr_obj.keys():
            child = zarr_obj[key]
            shape_dict[key] = zarr_apply(child, func)
        return shape_dict
    else:
        # If it's other type, return None
        return None

def _flat_dict(dct, prefix="/"):
    """
    Flatten nested dictionary to single-level dictionary, with nested paths as keys
    """
    flat_dict = {}
    for key, value in dct.items():
        if isinstance(value, dict):
            # If value is a dictionary, process recursively
            flat_dict.update(_flat_dict(value, prefix + key + "/"))
        else:
            # Otherwise add directly to results
            flat_dict[prefix + key] = value
    return flat_dict


def check_and_get_key_to_shape(root_group, output_root: Path):

    with open(output_root / "meta_info.pkl", "rb") as f:
        meta_info = pickle.load(f)
    shape_meta = meta_info["shape_meta"]
    key_to_shape = _flat_dict(shape_meta)

    print("Checking key_to_shape...")
    rows_num = -1
    for key, value in key_to_shape.items():
        if rows_num == -1:
            rows_num = value[0]
        assert value[0] == rows_num, f"key {key} has different number of rows: {value[0]} != {rows_num}"
    print("key_to_shape check passed, all keys have the same number of rows:", rows_num)
    return key_to_shape, rows_num

def process_meta_info(root, output_root: Path):
    meta_group = root["meta"]
    meta_info_json_raw = zarr_apply(meta_group, lambda x: x[()].tolist())
    shape_meta = zarr_apply(root["data"], lambda x: x.shape)
    meta_info_json_raw["initial_state"] = zarr_apply(root["data/initial_state"], lambda x: x[()].tolist())
    del shape_meta["initial_state"]
    meta_info_json = {
        "meta": meta_info_json_raw,
        "shape_meta": shape_meta,
    }
    with open(output_root / "meta_info.json", "w") as f:
        json.dump(meta_info_json, f, indent=4)
    meta_info = {}
    meta_info["meta"] = zarr_apply(meta_group, lambda x: x[()])
    meta_info["meta"]["initial_state"] = zarr_apply(root["data/initial_state"], lambda x: x[()])
    meta_info["shape_meta"] = shape_meta
    with open(output_root / "meta_info.pkl", "wb") as f:
        pickle.dump(meta_info, f)
    print("processed meta info and saved to", output_root / "meta_info.json", "and", "meta_info.pkl")

@ray.remote(num_cpus=1)
def process_subbatch(zarr_path, key_list, row_ids):
    """Process multiple rows of data in a sub-batch"""
    zarr_store = zarr.DirectoryStore(zarr_path)
    zarr_root = zarr.group(zarr_store)
    results = []
    
    for row_id in row_ids:
        data_one_row = {}
        for key in key_list:
            data = zarr_root["data"][key[1:]][row_id]
            data_one_row[key[1:]] = data
        data_one_row["row_id"] = row_id
        results.append(data_one_row)
    
    return results

def process_batch(zarr_path, key_list, start_idx, end_idx):
    """Process a complete batch, split it into multiple sub-batches for parallel processing"""
    # Read batch data
    ids = list(range(start_idx, end_idx))
    total_rows = len(ids)
    
    # Control maximum number of concurrent tasks
    max_concurrent_tasks = 64
    
    # Calculate sub-batch size, ensuring at most max_concurrent_tasks tasks
    subbatch_size = max(1, (total_rows + max_concurrent_tasks - 1) // max_concurrent_tasks)
    
    # Split into sub-batches
    subbatches = []
    for i in range(0, total_rows, subbatch_size):
        end = min(i + subbatch_size, total_rows)
        subbatches.append(ids[i:end])
    
    # Submit all sub-batch tasks
    futures = [process_subbatch.remote(zarr_path, key_list, subbatch) for subbatch in subbatches]
    
    # Get results and flatten in order
    results = []
    for result in ray.get(futures):
        results.extend(result)
    
    # Ensure results are sorted by row ID
    results.sort(key=lambda x: x["row_id"])
    
    return results

def zarr_to_lance_distributed(zarr_path, key_to_shape, rows_num, output_root, batch_size=1000):
    """Distributed reading of zarr data and writing to lance, using Ray Dataset for batch processing"""
    lance_path = str(output_root / "data.lance")
    
    # Calculate number of batches
    num_batches = (rows_num + batch_size - 1) // batch_size
    print(f"Processing {rows_num} rows of data, divided into {num_batches} batches, {batch_size} rows per batch")
    
    key_list = list(key_to_shape.keys())
    
    # Process first batch separately to create initial dataset
    print(f"Processing batch 1/{num_batches}...")
    first_batch = process_batch(zarr_path, key_list, 0, min(batch_size, rows_num))
    
    # Convert first batch to Ray Dataset
    ds = ray.data.from_items(first_batch)
    ds.write_lance(output_root / "data.lance", mode="overwrite")
    # Process remaining batches and append to dataset
    for i in tqdm.tgrange(1, num_batches):
        start_time = time.time()
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, rows_num)
        
        print(f"Processing batch {i+1}/{num_batches}...")
        batch_data = process_batch(zarr_path, key_list, start_idx, end_idx)
        
        # Convert to Ray Dataset and append
        batch_ds = ray.data.from_items(batch_data)
        batch_ds.write_lance(lance_path, mode="append")
        print(f"Batch {i+1} completed, took {time.time() - start_time:.2f} seconds")
    print(f"Dataset conversion completed, saved at: {lance_path}")

def main(zarr_path: str, address: str = "127.0.0.1:6379", output_dir: str = "", batch_size: int = 1000):
    ray.init(address)
    zarr_path = Path(zarr_path)
    output_dir = output_dir if output_dir else zarr_path.parent / f"{zarr_path.stem}_lance"
    output_root = Path(output_dir)
    zarr_store = zarr.DirectoryStore(zarr_path)
    zarr_root = zarr.group(zarr_store)
    output_root.mkdir(parents=True, exist_ok=True)
    process_meta_info(zarr_root, output_root)
    key_to_shape, rows_num = check_and_get_key_to_shape(zarr_root, output_root)
    zarr_to_lance_distributed(zarr_path, key_to_shape, rows_num, output_root, batch_size)
    
if __name__ =="__main__":
    tyro.cli(main)
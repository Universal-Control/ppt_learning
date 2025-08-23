import numpy as np
import time
import tyro
import os
import pickle

ignore_keys = ["row_id"]
def check_data_consistency(zarr_root, dsl, shape_meta, check_nums=1000):
    """
    Check zarr data consistency within specified index range
    
    Args:
        zarr_root: zarr root object
        start_idx: start index
        end_idx: end index
        
    Returns:
        list: list of dictionaries containing data for each row
    """
    indices = np.arange(len(dsl))
    np.random.shuffle(indices)
    if check_nums < len(indices):
        indices = indices[:check_nums]
    else:
        check_nums = len(indices)
    indices = np.sort(indices)
    keys = dsl.take([0]).column_names

    start_time = time.time()
    zarr_datas = {}
    for key in keys:
        if key in ignore_keys:
            continue
        zarr_datas[key] = zarr_root["data"][key][indices]
    print(f"Zarr data fetch time: {time.time() - start_time:.4f} seconds")
    start_time = time.time()
    dsl_datas = dsl.take(indices)
    print(f"Lance data fetch time: {time.time() - start_time:.4f} seconds")
    for key in keys:
        if key in ignore_keys:
            continue
        zarr_data = zarr_datas[key]
        dsl_data = dsl_datas.column(key)
        shape_key_lst = key.split("/")
        shape = shape_meta
        for shape_key in shape_key_lst:
            shape = shape[shape_key]
        if not np.array_equal(np.stack(dsl_data.to_numpy()).reshape(-1, *shape[1:]), zarr_data[indices]):
            print(f"Data mismatch for key: {key}")
            return False
    print("All data matches!")
    return True

def main(zarr_path: str, lance_path: str, check_nums: int = 1000):
    import zarr
    import lance
    from pathlib import Path

    zarr_store = zarr.DirectoryStore(zarr_path)
    zarr_root = zarr.group(zarr_store)
    
    # Load Lance dataset
    dsl = lance.dataset(os.path.join(lance_path, "data.lance"))
    with open(os.path.join(lance_path, "meta_info.pkl"), "rb") as f:
        shape_meta = pickle.load(f)["shape_meta"]

    # Check data consistency
    if not check_data_consistency(zarr_root, dsl, shape_meta, check_nums):
        print("Data inconsistency found!")
    else:
        print("Data consistency check passed!")

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
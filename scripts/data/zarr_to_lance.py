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
    获取zarr对象的shape结构，返回嵌套字典
    
    Args:
        zarr_obj: zarr.Group 或 zarr.Array 对象
        
    Returns:
        dict: 嵌套字典，最终的value是shape tuple
    """
    if isinstance(zarr_obj, zarr.Array):
        # 如果是数组，直接返回shape
        return func(zarr_obj, *args, **kwargs)
    elif isinstance(zarr_obj, zarr.Group):
        # 如果是组，递归处理所有子项
        shape_dict = {}
        for key in zarr_obj.keys():
            child = zarr_obj[key]
            shape_dict[key] = zarr_apply(child, func)
        return shape_dict
    else:
        # 如果是其他类型，返回None
        return None

def _flat_dict(dct, prefix="/"):
    """
    将嵌套字典展平为一层字典，key为嵌套路径
    """
    flat_dict = {}
    for key, value in dct.items():
        if isinstance(value, dict):
            # 如果value是字典，递归处理
            flat_dict.update(_flat_dict(value, prefix + key + "/"))
        else:
            # 否则直接添加到结果中
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
    """处理一个子批次的多行数据"""
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
    """处理一个完整批次，将其分割为多个子批次并行处理"""
    # 读取batch数据
    ids = list(range(start_idx, end_idx))
    total_rows = len(ids)
    
    # 控制并发任务数量的最大值
    max_concurrent_tasks = 64
    
    # 计算每个子批次的大小，确保最多有max_concurrent_tasks个任务
    subbatch_size = max(1, (total_rows + max_concurrent_tasks - 1) // max_concurrent_tasks)
    
    # 分割成子批次
    subbatches = []
    for i in range(0, total_rows, subbatch_size):
        end = min(i + subbatch_size, total_rows)
        subbatches.append(ids[i:end])
    
    # 提交所有子批次任务
    futures = [process_subbatch.remote(zarr_path, key_list, subbatch) for subbatch in subbatches]
    
    # 获取结果并按顺序展平
    results = []
    for result in ray.get(futures):
        results.extend(result)
    
    # 确保结果按行ID排序
    results.sort(key=lambda x: x["row_id"])
    
    return results

def zarr_to_lance_distributed(zarr_path, key_to_shape, rows_num, output_root, batch_size=1000):
    """分布式读取zarr数据并写入lance，使用Ray Dataset分批处理"""
    lance_path = str(output_root / "data.lance")
    
    # 计算批次数
    num_batches = (rows_num + batch_size - 1) // batch_size
    print(f"处理 {rows_num} 行数据，分 {num_batches} 个批次，每批 {batch_size} 行")
    
    key_list = list(key_to_shape.keys())
    
    # 第一个批次单独处理，创建初始数据集
    print(f"处理第1/{num_batches}批次...")
    first_batch = process_batch(zarr_path, key_list, 0, min(batch_size, rows_num))
    
    # 转换第一个批次为Ray Dataset
    ds = ray.data.from_items(first_batch)
    ds.write_lance(output_root / "data.lance", mode="overwrite")
    # 处理剩余批次并追加到数据集
    for i in tqdm.tgrange(1, num_batches):
        start_time = time.time()
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, rows_num)
        
        print(f"处理第{i+1}/{num_batches}批次...")
        batch_data = process_batch(zarr_path, key_list, start_idx, end_idx)
        
        # 转换为Ray Dataset并追加
        batch_ds = ray.data.from_items(batch_data)
        batch_ds.write_lance(lance_path, mode="append")
        print(f"批次 {i+1} 处理完成，耗时 {time.time() - start_time:.2f} 秒")
    print(f"数据集转换完成，保存在: {lance_path}")

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
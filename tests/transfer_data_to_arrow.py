#!/usr/bin/env python3
"""
使用 Ray Data 将 Zarr 数据转换为 Parquet 格式
优化版本 - 修复生成器长度问题和其他内存优化
Author: xshenhan
Date: 2025-06-07
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Iterator
import argparse
from datetime import datetime

import ray
import ray.data
import zarr
import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
import logging
import traceback
from contextlib import contextmanager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ZarrChunkIterator:
    """Zarr 数据块迭代器 - 支持错误恢复和流式处理"""
    
    def __init__(self, zarr_path: str, chunk_strategy: List[Dict[str, Any]], 
                 continue_on_error: bool = True, max_retries: int = 3):
        self.zarr_path = zarr_path
        self.chunk_strategy = chunk_strategy
        self.continue_on_error = continue_on_error
        self.max_retries = max_retries
        self.error_log = []
        self._chunks = None
    
    def __len__(self):
        """返回块的数量"""
        return len(self.chunk_strategy)
    
    def __iter__(self):
        """迭代器，yield 每个数据块 - 支持错误恢复"""
        if self._chunks is None:
            self._chunks = list(self._generate_chunks())
        return iter(self._chunks)
    
    def _generate_chunks(self) -> Iterator[Dict[str, Any]]:
        """生成数据块的内部方法"""
        zarr_root = None
        
        try:
            zarr_root = zarr.open(self.zarr_path, mode='r')
        except Exception as e:
            logger.error(f"无法打开 Zarr 文件 {self.zarr_path}: {e}")
            if not self.continue_on_error:
                raise
            return
        
        for chunk_info in self.chunk_strategy:
            chunk_id = chunk_info.get('chunk_id', 'unknown')
            retry_count = 0
            
            while retry_count <= self.max_retries:
                try:
                    data_path = chunk_info['data_path']
                    meta_path = chunk_info.get('meta_path')
                    slice_info = chunk_info.get('slice_info')
                    
                    # 流式提取数据，避免全部加载到内存
                    if slice_info:
                        zarr_array = zarr_root[data_path]
                        # 检查数组大小，如果太大则进一步分割
                        if hasattr(slice_info, 'start') and hasattr(slice_info, 'stop'):
                            slice_size = (slice_info.stop - slice_info.start) * zarr_array.itemsize * np.prod(zarr_array.shape[1:])
                            if slice_size > 50 * 1024 * 1024:  # 50MB 限制
                                logger.warning(f"数据块 {chunk_id} 太大 ({slice_size/1024/1024:.1f}MB)，将进一步分割")
                                # 这里可以实现进一步分割逻辑
                        
                        data = zarr_array[slice_info]
                    else:
                        # 对于大数组，分批读取
                        zarr_array = zarr_root[data_path]
                        if zarr_array.nbytes > 100 * 1024 * 1024:  # 100MB
                            logger.warning(f"大数组 {data_path} ({zarr_array.nbytes/1024/1024:.1f}MB)，将分批处理")
                        data = zarr_array[:]
                    
                    # 提取元数据
                    meta_data = {}
                    if meta_path and meta_path in zarr_root:
                        try:
                            meta_data = dict(zarr_root[meta_path].attrs)
                        except Exception as meta_e:
                            logger.warning(f"无法读取元数据 {meta_path}: {meta_e}")
                    
                    # 预处理数据以避免 Arrow 维度问题
                    processed_data = self._preprocess_data(data, chunk_id)
                    
                    # 生成单个数据块记录
                    yield {
                        "chunk_id": chunk_id,
                        "data_path": data_path,
                        "data": processed_data,
                        "meta_data": meta_data,
                        "slice_info": str(slice_info) if slice_info else None,
                        "original_shape": data.shape if hasattr(data, 'shape') else None,
                        "data_type": str(data.dtype) if hasattr(data, 'dtype') else None
                    }
                    break  # 成功处理，跳出重试循环
                    
                except Exception as e:
                    retry_count += 1
                    error_msg = f"处理块 {chunk_id} (重试 {retry_count}/{self.max_retries}) 时出错: {e}"
                    logger.error(error_msg)
                    self.error_log.append({
                        "chunk_id": chunk_id,
                        "error": str(e),
                        "retry_count": retry_count,
                        "traceback": traceback.format_exc()
                    })
                    
                    if retry_count > self.max_retries:
                        if self.continue_on_error:
                            logger.warning(f"跳过块 {chunk_id}，已达最大重试次数")
                            # 生成一个空的占位符块
                            yield {
                                "chunk_id": chunk_id,
                                "data_path": chunk_info['data_path'],
                                "data": np.array([]),  # 空数组
                                "meta_data": {"error": str(e)},
                                "slice_info": str(slice_info) if slice_info else None,
                                "original_shape": None,
                                "data_type": "error"
                            }
                            break
                        else:
                            raise
                    else:
                        time.sleep(1)  # 重试前等待
    
    def _preprocess_data(self, data: np.ndarray, chunk_id: str) -> np.ndarray:
        """预处理数据以避免 Arrow 维度不一致问题"""
        if not isinstance(data, np.ndarray):
            return data
        
        # 确保所有数据块具有一致的维度
        if data.ndim == 0:
            # 标量转为1D数组
            return np.array([data])
        elif data.ndim == 1:
            # 1D数组保持不变
            return data
        elif data.ndim >= 2:
            # 多维数组展平为2D (samples, features)
            if data.ndim == 2:
                return data
            else:
                # 保留第一维度作为样本数，展平其他维度
                new_shape = (data.shape[0], -1)
                reshaped = data.reshape(new_shape)
                logger.debug(f"块 {chunk_id}: 重塑 {data.shape} -> {reshaped.shape}")
                return reshaped
        
        return data


def create_zarr_dataset_from_chunks(chunk_iterator: ZarrChunkIterator) -> ray.data.Dataset:
    """从 Zarr 块迭代器创建 Ray Dataset"""
    try:
        # 将迭代器转换为列表以获得长度
        chunks_list = list(chunk_iterator)
        logger.info(f"成功加载 {len(chunks_list)} 个数据块定义")
        
        # 使用列表创建 dataset
        dataset = ray.data.from_items(chunks_list)
        return dataset
        
    except Exception as e:
        logger.error(f"创建 Ray Dataset 时出错: {e}")
        raise


def normalize_data_dimensions(batch_data: List[np.ndarray]) -> List[np.ndarray]:
    """标准化批次中所有数据的维度"""
    if not batch_data:
        return batch_data
    
    # 找出最大维度数
    valid_data = [data for data in batch_data if isinstance(data, np.ndarray) and data.size > 0]
    if not valid_data:
        return batch_data
    
    max_dims = max(data.ndim for data in valid_data)
    
    normalized_data = []
    for data in batch_data:
        if not isinstance(data, np.ndarray) or data.size == 0:
            normalized_data.append(data)
            continue
            
        if data.ndim < max_dims:
            # 如果维度不足，添加新轴
            if data.ndim == 1 and max_dims == 2:
                # 1D -> 2D: (n,) -> (n, 1)
                data = data.reshape(-1, 1)
            elif data.ndim == 0 and max_dims >= 1:
                # 标量 -> 1D
                data = np.array([data])
                if max_dims == 2:
                    data = data.reshape(1, 1)
        elif data.ndim > 2:
            # 超过2D的数组展平
            data = data.reshape(data.shape[0], -1)
        
        normalized_data.append(data)
    
    return normalized_data


def prepare_data_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ray Data UDF: 将 Zarr 数据块转换为训练友好的格式
    增强错误处理和内存优化
    """
    try:
        # 处理批次数据 - 确保是列表格式
        if isinstance(batch["chunk_id"], list):
            chunk_ids = batch["chunk_id"]
            data_list = batch["data"]
            meta_data_list = batch["meta_data"]
            data_paths = batch.get("data_path", [])
        else:
            chunk_ids = [batch["chunk_id"]]
            data_list = [batch["data"]]
            meta_data_list = [batch["meta_data"]]
            data_paths = [batch.get("data_path", "")]
        
        # 跳过错误块
        valid_data = []
        valid_chunk_ids = []
        valid_meta_data = []
        valid_data_paths = []
        
        for i, (chunk_id, data, meta_data) in enumerate(zip(chunk_ids, data_list, meta_data_list)):
            if isinstance(data, np.ndarray) and data.size > 0:
                valid_data.append(data)
                valid_chunk_ids.append(chunk_id)
                valid_meta_data.append(meta_data)
                valid_data_paths.append(data_paths[i] if i < len(data_paths) else "")
            else:
                logger.warning(f"跳过空或错误的数据块: {chunk_id}")
        
        if not valid_data:
            logger.warning("批次中没有有效数据")
            return pd.DataFrame()
        
        # 标准化数据维度
        normalized_data = normalize_data_dimensions(valid_data)
        
        # 合并所有有效数据
        all_dataframes = []
        
        for chunk_id, data, meta_data, data_path in zip(valid_chunk_ids, normalized_data, valid_meta_data, valid_data_paths):
            try:
                # 处理多维数组
                if isinstance(data, np.ndarray):
                    if data.ndim > 2:
                        original_shape = data.shape
                        # 保留第一维作为样本数，展平其他维度
                        data = data.reshape(data.shape[0], -1)
                        logger.debug(f"重塑数据从 {original_shape} 到 {data.shape}")
                    
                    # 创建 DataFrame
                    if data.ndim == 1:
                        df = pd.DataFrame({"value": data})
                    elif data.ndim == 2:
                        # 为特征创建列名
                        feature_names = [f"feature_{i:04d}" for i in range(data.shape[1])]
                        df = pd.DataFrame(data, columns=feature_names)
                    else:
                        # 应该不会到这里，但以防万一
                        df = pd.DataFrame({"value": data.flatten()})
                else:
                    # 处理非数组数据
                    df = pd.DataFrame({"value": [data]})
                
                # 添加元数据列
                if meta_data and not meta_data.get("error"):
                    for key, value in meta_data.items():
                        if isinstance(value, (str, int, float, bool)):
                            df[f"meta_{key}"] = value
                        elif isinstance(value, (list, tuple)) and len(value) == len(df):
                            df[f"meta_{key}"] = value
                
                # 添加标识列
                df["chunk_id"] = chunk_id
                df["row_id"] = range(len(df))
                df["data_path"] = data_path
                
                all_dataframes.append(df)
                
            except Exception as e:
                logger.error(f"处理单个数据块 {chunk_id} 时出错: {e}")
                continue
        
        if not all_dataframes:
            logger.warning("没有成功处理的数据框")
            return pd.DataFrame()
        
        # 合并所有数据框
        try:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # 确保所有列都有一致的数据类型
            combined_df = harmonize_column_types(combined_df)
            
            return combined_df
            
        except Exception as e:
            logger.error(f"合并数据框时出错: {e}")
            # 返回第一个有效的数据框
            return all_dataframes[0]
        
    except Exception as e:
        logger.error(f"处理数据批次时出错: {e}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        # 返回空表
        return pd.DataFrame()


def harmonize_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """统一数据框中列的数据类型"""
    try:
        # 处理数值列的类型不一致问题
        for col in df.columns:
            if df[col].dtype == 'object':
                # 尝试转换为数值类型
                try:
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_col.isna().all():
                        df[col] = numeric_col
                except:
                    pass
        
        return df
    except Exception as e:
        logger.warning(f"类型统一化失败: {e}")
        return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """优化 DataFrame 的数据类型以节省内存"""
    try:
        for col in df.columns:
            if df[col].dtype == 'object':
                continue
                
            if df[col].dtype in ['int64']:
                col_min, col_max = df[col].min(), df[col].max()
                if pd.isna(col_min) or pd.isna(col_max):
                    continue
                    
                if col_min >= 0:
                    if col_max < 255:
                        df[col] = df[col].astype('uint8')
                    elif col_max < 65535:
                        df[col] = df[col].astype('uint16')
                    elif col_max < 4294967295:
                        df[col] = df[col].astype('uint32')
                else:
                    if col_min >= -128 and col_max < 127:
                        df[col] = df[col].astype('int8')
                    elif col_min >= -32768 and col_max < 32767:
                        df[col] = df[col].astype('int16')
                    elif col_min >= -2147483648 and col_max < 2147483647:
                        df[col] = df[col].astype('int32')
            
            elif df[col].dtype in ['float64']:
                # 检查是否可以安全转换为 float32
                if (df[col].min() >= np.finfo(np.float32).min and 
                    df[col].max() <= np.finfo(np.float32).max):
                    df[col] = df[col].astype('float32')
    except Exception as e:
        logger.warning(f"数据类型优化失败: {e}")
    
    return df


def analyze_zarr_structure(zarr_path: str) -> Dict[str, Any]:
    """分析 Zarr 数据结构"""
    zarr_root = zarr.open(zarr_path, mode='r')
    
    structure = {
        "meta_keys": [],
        "data_keys": [],
        "total_size": 0,
        "data_shapes": {},
        "data_dtypes": {},
        "chunk_strategies": []
    }
    
    def explore_group(group, path=""):
        for key in group.keys():
            full_path = f"{path}/{key}" if path else key
            item = group[key]
            
            if isinstance(item, zarr.Group):
                explore_group(item, full_path)
            elif isinstance(item, zarr.Array):
                if full_path.startswith("meta"):
                    structure["meta_keys"].append(full_path)
                elif full_path.startswith("data"):
                    structure["data_keys"].append(full_path)
                    structure["data_shapes"][full_path] = item.shape
                    structure["data_dtypes"][full_path] = str(item.dtype)
                    structure["total_size"] += item.nbytes
    
    explore_group(zarr_root)
    return structure


def create_chunk_strategy(
    zarr_structure: Dict[str, Any], 
    target_chunk_size_mb: int = 64,
    num_workers: int = None
) -> List[Dict[str, Any]]:
    """创建优化的数据块分割策略"""
    
    if num_workers is None:
        num_workers = int(ray.cluster_resources().get("CPU", 1))
    
    chunks = []
    chunk_id = 0
    
    for data_path in zarr_structure["data_keys"]:
        shape = zarr_structure["data_shapes"][data_path]
        dtype = zarr_structure["data_dtypes"][data_path]
        
        if len(shape) == 0:
            continue
        
        # 计算每个元素的字节数
        try:
            element_size = np.dtype(dtype).itemsize
        except:
            element_size = 8  # 默认假设 float64
        
        # 计算每行的字节数
        row_size = element_size * np.prod(shape[1:]) if len(shape) > 1 else element_size
        
        # 计算每个块的最大行数 - 更保守的内存使用
        target_chunk_bytes = target_chunk_size_mb * 1024 * 1024
        max_rows_per_chunk = max(1, target_chunk_bytes // row_size)  # 移除 *2 的缓冲
        
        # 确保块数量不会太多
        total_rows = shape[0]
        min_rows_per_chunk = max(1, total_rows // (num_workers * 2))  # 从8改为2，减少块数量
        max_rows_per_chunk = max(min_rows_per_chunk, max_rows_per_chunk)
        
        # 创建数据块
        for start_idx in range(0, total_rows, max_rows_per_chunk):
            end_idx = min(start_idx + max_rows_per_chunk, total_rows)
            
            chunk_info = {
                "chunk_id": chunk_id,
                "data_path": data_path,
                "slice_info": slice(start_idx, end_idx),
                "estimated_rows": end_idx - start_idx,
                "estimated_size_mb": (end_idx - start_idx) * row_size / 1024 / 1024,
                "meta_path": None
            }
            
            # 查找对应的元数据路径
            data_key = data_path.split("/")[-1]
            for meta_path in zarr_structure["meta_keys"]:
                if data_key in meta_path:
                    chunk_info["meta_path"] = meta_path
                    break
            
            chunks.append(chunk_info)
            chunk_id += 1
    
    return chunks


@contextmanager
def memory_monitor():
    """内存监控上下文管理器"""
    try:
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        yield mem_before
        mem_after = process.memory_info().rss / 1024 / 1024
        logger.info(f"内存使用: {mem_before:.1f}MB -> {mem_after:.1f}MB (增加: {mem_after-mem_before:.1f}MB)")
    except ImportError:
        yield 0


def main():
    parser = argparse.ArgumentParser(description="Convert Zarr to Parquet using Ray Data")
    parser.add_argument("zarr_path", help="Path to input Zarr file/directory")
    parser.add_argument("--output-dir", help="Output directory (default: same as input)")
    parser.add_argument("--num-workers", type=int, help="Number of Ray workers")
    parser.add_argument("--chunk-size-mb", type=int, default=2048,  # 从32改为1024
                    help="Target chunk size in MB")
    parser.add_argument("--block-size-mb", type=int, default=2048,  # 从128改为2048
                    help="Ray Data block size in MB")
    parser.add_argument("--compression", default="snappy", 
                       choices=["snappy", "gzip", "brotli", "lz4", "zstd"],
                       help="Parquet compression algorithm")
    parser.add_argument("--ray-address", help="Ray cluster address")
    parser.add_argument("--parallelism", type=int, help="Ray Data parallelism override", default=None)
    parser.add_argument("--continue-on-error", action="store_true", default=True,
                       help="Continue processing on errors")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retries per chunk")
    
    args = parser.parse_args()
    
    # 设置输出目录
    zarr_path = Path(args.zarr_path)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = zarr_path.parent / f"{zarr_path.stem}_parquet"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化 Ray
    ray_init_kwargs = {"ignore_reinit_error": True}
    if args.ray_address:
        ray_init_kwargs["address"] = args.ray_address
    if args.num_workers:
        ray_init_kwargs["num_cpus"] = args.num_workers
    
    ray.init(**ray_init_kwargs)
    
    try:
        with memory_monitor() as initial_memory:
            logger.info(f"开始分析 Zarr 结构: {zarr_path}")
            zarr_structure = analyze_zarr_structure(str(zarr_path))
            
            logger.info(f"发现数据键: {zarr_structure['data_keys']}")
            logger.info(f"发现元数据键: {zarr_structure['meta_keys']}")
            logger.info(f"总数据大小: {zarr_structure['total_size'] / 1024 / 1024:.2f} MB")
            
            # 创建分块策略
            chunks = create_chunk_strategy(
                zarr_structure, 
                target_chunk_size_mb=args.chunk_size_mb,
                num_workers=args.num_workers or int(ray.cluster_resources().get("CPU", 1))
            )
            
            logger.info(f"创建了 {len(chunks)} 个数据块")
            total_estimated_size = sum(chunk["estimated_size_mb"] for chunk in chunks)
            logger.info(f"预估总处理大小: {total_estimated_size:.2f} MB")
            
            # 创建 Zarr 数据迭代器
            logger.info("创建 Zarr 数据迭代器...")
            chunk_iterator = ZarrChunkIterator(
                str(zarr_path), 
                chunks, 
                continue_on_error=args.continue_on_error,
                max_retries=args.max_retries
            )
            
            # 创建 Ray Dataset
            logger.info("创建 Ray Dataset...")
            dataset = create_zarr_dataset_from_chunks(chunk_iterator)
            dataset = dataset.prefetch(num_blocks=4)  # 添加预取
            
            # 设置 Ray Data 配置
            ctx = ray.data.DataContext.get_current()
            ctx.execution_options.preserve_order = True
            if args.block_size_mb:
                ctx.target_max_block_size = args.block_size_mb * 1024 * 1024
            
            # 设置并行度
            if args.parallelism:
                dataset = dataset.repartition(args.parallelism)
            else:
                # 自动设置合理的并行度
                num_cpus = int(ray.cluster_resources().get("CPU", 1))
                optimal_parallelism = min(len(chunks), num_cpus * 4)
                dataset = dataset.repartition(optimal_parallelism)
            
            logger.info(f"Dataset 并行度: {dataset.num_blocks()}")
            
            # 应用数据转换
            logger.info("开始数据转换...")
            start_time = time.time()
            
            # 使用更小的批次大小来避免内存问题
            processed_dataset = dataset.map_batches(
                prepare_data_batch,
                batch_format="pandas",
                batch_size=8,  # 从1改为8，一次处理多个块
                num_cpus=2,    # 从1改为2，每个任务使用更多CPU
                zero_copy_batch=True,
            )
            
            # 过滤空的数据框
            def filter_empty_dataframes(batch):
                """过滤空的数据框"""
                if isinstance(batch, pd.DataFrame) and len(batch) > 0:
                    return batch
                else:
                    return pd.DataFrame()
            
            final_dataset = processed_dataset.map_batches(
                filter_empty_dataframes,
                batch_format="pandas",
                batch_size=None,
                zero_copy_batch=True
            )
            
            # 写入 Parquet 文件
            logger.info(f"写入 Parquet 文件到: {output_dir}")
            
            # Ray Data 的 write_parquet 方法
            final_dataset.write_parquet(
                str(output_dir),
                compression=args.compression,
                try_create_dir=True,
                row_group_size=100000,  # 从25000改为100000，增加行组大小
                use_threads=True,
                num_rows_per_file=1000000,  # 添加此参数，增加每个文件的行数
            )
            
            conversion_time = time.time() - start_time
            logger.info(f"转换完成! 用时: {conversion_time:.2f} 秒")
            
            # 记录错误统计
            if hasattr(chunk_iterator, 'error_log') and chunk_iterator.error_log:
                logger.warning(f"处理过程中发生 {len(chunk_iterator.error_log)} 个错误")
                error_file = output_dir / "error_log.json"
                with open(error_file, 'w') as f:
                    json.dump(chunk_iterator.error_log, f, indent=2)
                logger.info(f"错误日志保存到: {error_file}")
            
            # 创建数据集元数据
            dataset_metadata = {
                "conversion_date": datetime.now().isoformat(),
                "source_zarr": str(zarr_path),
                "output_directory": str(output_dir),
                "total_chunks": len(chunks),
                "conversion_time_seconds": conversion_time,
                "compression": args.compression,
                "ray_data_parallelism": final_dataset.num_blocks(),
                "zarr_structure": zarr_structure,
                "chunk_strategy_summary": {
                    "total_chunks": len(chunks),
                    "avg_chunk_size_mb": total_estimated_size / len(chunks) if chunks else 0,
                    "total_estimated_size_mb": total_estimated_size
                },
                "error_count": len(chunk_iterator.error_log) if hasattr(chunk_iterator, 'error_log') else 0,
                "continue_on_error": args.continue_on_error,
                "max_retries": args.max_retries
            }
            
            # 保存元数据
            metadata_file = output_dir / "dataset_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(dataset_metadata, f, indent=2)
            
            # 验证输出
            parquet_files = list(output_dir.glob("*.parquet"))
            total_size = sum(f.stat().st_size for f in parquet_files)
            
            logger.info("=" * 60)
            logger.info("转换完成统计:")
            logger.info(f"输出目录: {output_dir}")
            logger.info(f"生成文件数: {len(parquet_files)}")
            logger.info(f"总输出大小: {total_size / 1024 / 1024:.2f} MB")
            if zarr_structure['total_size'] > 0:
                logger.info(f"压缩比: {zarr_structure['total_size'] / total_size:.2f}x")
            logger.info(f"转换速度: {zarr_structure['total_size'] / 1024 / 1024 / conversion_time:.2f} MB/s")
            logger.info(f"元数据文件: {metadata_file}")
            if hasattr(chunk_iterator, 'error_log') and chunk_iterator.error_log:
                logger.info(f"错误数量: {len(chunk_iterator.error_log)} (详见 error_log.json)")
            logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"转换过程出错: {e}")
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
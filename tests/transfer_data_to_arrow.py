#!/usr/bin/env python3
"""
Convert Zarr data to Parquet format using Ray Data
Optimized version - Fixed generator length issues and other memory optimizations
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

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ZarrChunkIterator:
    """Zarr data chunk iterator - supports error recovery and streaming"""

    def __init__(
        self,
        zarr_path: str,
        chunk_strategy: List[Dict[str, Any]],
        continue_on_error: bool = True,
        max_retries: int = 3,
    ):
        self.zarr_path = zarr_path
        self.chunk_strategy = chunk_strategy
        self.continue_on_error = continue_on_error
        self.max_retries = max_retries
        self.error_log = []
        self._chunks = None

    def __len__(self):
        """Return the number of chunks"""
        return len(self.chunk_strategy)

    def __iter__(self):
        """Iterator that yields each data chunk - supports error recovery"""
        if self._chunks is None:
            self._chunks = list(self._generate_chunks())
        return iter(self._chunks)

    def _generate_chunks(self) -> Iterator[Dict[str, Any]]:
        """Internal method to generate data chunks"""
        zarr_root = None

        try:
            zarr_root = zarr.open(self.zarr_path, mode="r")
        except Exception as e:
            logger.error(f"Unable to open Zarr file {self.zarr_path}: {e}")
            if not self.continue_on_error:
                raise
            return

        for chunk_info in self.chunk_strategy:
            chunk_id = chunk_info.get("chunk_id", "unknown")
            retry_count = 0

            while retry_count <= self.max_retries:
                try:
                    data_path = chunk_info["data_path"]
                    meta_path = chunk_info.get("meta_path")
                    slice_info = chunk_info.get("slice_info")

                    # Stream data extraction, avoid loading everything into memory
                    if slice_info:
                        zarr_array = zarr_root[data_path]
                        # Check array size, split further if too large
                        if hasattr(slice_info, "start") and hasattr(slice_info, "stop"):
                            slice_size = (
                                (slice_info.stop - slice_info.start)
                                * zarr_array.itemsize
                                * np.prod(zarr_array.shape[1:])
                            )
                            if slice_size > 50 * 1024 * 1024:  # 50MB limit
                                logger.warning(
                                    f"Data chunk {chunk_id} is too large ({slice_size/1024/1024:.1f}MB), will split further"
                                )
                                # Further splitting logic can be implemented here

                        data = zarr_array[slice_info]
                    else:
                        # For large arrays, read in batches
                        zarr_array = zarr_root[data_path]
                        if zarr_array.nbytes > 100 * 1024 * 1024:  # 100MB
                            logger.warning(
                                f"Large array {data_path} ({zarr_array.nbytes/1024/1024:.1f}MB), will process in batches"
                            )
                        data = zarr_array[:]

                    # Extract metadata
                    meta_data = {}
                    if meta_path and meta_path in zarr_root:
                        try:
                            meta_data = dict(zarr_root[meta_path].attrs)
                        except Exception as meta_e:
                            logger.warning(
                                f"Unable to read metadata {meta_path}: {meta_e}"
                            )

                    # Preprocess data to avoid Arrow dimension issues
                    processed_data = self._preprocess_data(data, chunk_id)

                    # Generate single data chunk record
                    yield {
                        "chunk_id": chunk_id,
                        "data_path": data_path,
                        "data": processed_data,
                        "meta_data": meta_data,
                        "slice_info": str(slice_info) if slice_info else None,
                        "original_shape": (
                            data.shape if hasattr(data, "shape") else None
                        ),
                        "data_type": (
                            str(data.dtype) if hasattr(data, "dtype") else None
                        ),
                    }
                    break  # Successfully processed, exit retry loop

                except Exception as e:
                    retry_count += 1
                    error_msg = f"Error processing chunk {chunk_id} (retry {retry_count}/{self.max_retries}): {e}"
                    logger.error(error_msg)
                    self.error_log.append(
                        {
                            "chunk_id": chunk_id,
                            "error": str(e),
                            "retry_count": retry_count,
                            "traceback": traceback.format_exc(),
                        }
                    )

                    if retry_count > self.max_retries:
                        if self.continue_on_error:
                            logger.warning(
                                f"Skipping chunk {chunk_id}, maximum retry attempts reached"
                            )
                            # Generate an empty placeholder chunk
                            yield {
                                "chunk_id": chunk_id,
                                "data_path": chunk_info["data_path"],
                                "data": np.array([]),  # Empty array
                                "meta_data": {"error": str(e)},
                                "slice_info": str(slice_info) if slice_info else None,
                                "original_shape": None,
                                "data_type": "error",
                            }
                            break
                        else:
                            raise
                    else:
                        time.sleep(1)  # Wait before retry

    def _preprocess_data(self, data: np.ndarray, chunk_id: str) -> np.ndarray:
        """Preprocess data to avoid Arrow dimension inconsistency issues"""
        if not isinstance(data, np.ndarray):
            return data

        # Ensure all data chunks have consistent dimensions
        if data.ndim == 0:
            # Convert scalar to 1D array
            return np.array([data])
        elif data.ndim == 1:
            # Keep 1D array unchanged
            return data
        elif data.ndim >= 2:
            # Flatten multi-dimensional array to 2D (samples, features)
            if data.ndim == 2:
                return data
            else:
                # Keep first dimension as sample count, flatten other dimensions
                new_shape = (data.shape[0], -1)
                reshaped = data.reshape(new_shape)
                logger.debug(
                    f"Chunk {chunk_id}: reshaped {data.shape} -> {reshaped.shape}"
                )
                return reshaped

        return data


def create_zarr_dataset_from_chunks(
    chunk_iterator: ZarrChunkIterator,
) -> ray.data.Dataset:
    """Create Ray Dataset from Zarr chunk iterator"""
    try:
        # Convert iterator to list to get length
        chunks_list = list(chunk_iterator)
        logger.info(f"Successfully loaded {len(chunks_list)} data chunk definitions")

        # Create dataset using list
        dataset = ray.data.from_items(chunks_list)
        return dataset

    except Exception as e:
        logger.error(f"Error creating Ray Dataset: {e}")
        raise


def normalize_data_dimensions(batch_data: List[np.ndarray]) -> List[np.ndarray]:
    """Normalize dimensions of all data in the batch"""
    if not batch_data:
        return batch_data

    # Find maximum number of dimensions
    valid_data = [
        data for data in batch_data if isinstance(data, np.ndarray) and data.size > 0
    ]
    if not valid_data:
        return batch_data

    max_dims = max(data.ndim for data in valid_data)

    normalized_data = []
    for data in batch_data:
        if not isinstance(data, np.ndarray) or data.size == 0:
            normalized_data.append(data)
            continue

        if data.ndim < max_dims:
            # If dimensions are insufficient, add new axes
            if data.ndim == 1 and max_dims == 2:
                # 1D -> 2D: (n,) -> (n, 1)
                data = data.reshape(-1, 1)
            elif data.ndim == 0 and max_dims >= 1:
                # Scalar -> 1D
                data = np.array([data])
                if max_dims == 2:
                    data = data.reshape(1, 1)
        elif data.ndim > 2:
            # Flatten arrays beyond 2D
            data = data.reshape(data.shape[0], -1)

        normalized_data.append(data)

    return normalized_data


def prepare_data_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ray Data UDF: Convert Zarr data chunks to training-friendly format
    Enhanced error handling and memory optimization
    """
    try:
        # Process batch data - ensure list format
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

        # Skip error chunks
        valid_data = []
        valid_chunk_ids = []
        valid_meta_data = []
        valid_data_paths = []

        for i, (chunk_id, data, meta_data) in enumerate(
            zip(chunk_ids, data_list, meta_data_list)
        ):
            if isinstance(data, np.ndarray) and data.size > 0:
                valid_data.append(data)
                valid_chunk_ids.append(chunk_id)
                valid_meta_data.append(meta_data)
                valid_data_paths.append(data_paths[i] if i < len(data_paths) else "")
            else:
                logger.warning(f"Skipping empty or error chunk: {chunk_id}")

        if not valid_data:
            logger.warning("No valid data in batch")
            return pd.DataFrame()

        # Normalize data dimensions
        normalized_data = normalize_data_dimensions(valid_data)

        # Merge all valid data
        all_dataframes = []

        for chunk_id, data, meta_data, data_path in zip(
            valid_chunk_ids, normalized_data, valid_meta_data, valid_data_paths
        ):
            try:
                # Process multi-dimensional arrays
                if isinstance(data, np.ndarray):
                    if data.ndim > 2:
                        original_shape = data.shape
                        # Keep first dimension as sample count, flatten other dimensions
                        data = data.reshape(data.shape[0], -1)
                        logger.debug(
                            f"Reshaped data from {original_shape} to {data.shape}"
                        )

                    # Create DataFrame
                    if data.ndim == 1:
                        df = pd.DataFrame({"value": data})
                    elif data.ndim == 2:
                        # Create column names for features
                        feature_names = [
                            f"feature_{i:04d}" for i in range(data.shape[1])
                        ]
                        df = pd.DataFrame(data, columns=feature_names)
                    else:
                        # Should not reach here, but just in case
                        df = pd.DataFrame({"value": data.flatten()})
                else:
                    # Handle non-array data
                    df = pd.DataFrame({"value": [data]})

                # Add metadata columns
                if meta_data and not meta_data.get("error"):
                    for key, value in meta_data.items():
                        if isinstance(value, (str, int, float, bool)):
                            df[f"meta_{key}"] = value
                        elif isinstance(value, (list, tuple)) and len(value) == len(df):
                            df[f"meta_{key}"] = value

                # Add identifier columns
                df["chunk_id"] = chunk_id
                df["row_id"] = range(len(df))
                df["data_path"] = data_path

                all_dataframes.append(df)

            except Exception as e:
                logger.error(f"Error processing single data chunk {chunk_id}: {e}")
                continue

        if not all_dataframes:
            logger.warning("No successfully processed data frames")
            return pd.DataFrame()

        # Merge all data frames
        try:
            combined_df = pd.concat(all_dataframes, ignore_index=True)

            # Ensure all columns have consistent data types
            combined_df = harmonize_column_types(combined_df)

            return combined_df

        except Exception as e:
            logger.error(f"Error merging data frames: {e}")
            # Return first valid data frame
            return all_dataframes[0]

    except Exception as e:
        logger.error(f"Error processing data batch: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        # Return empty table
        return pd.DataFrame()


def harmonize_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize data types of columns in the data frame"""
    try:
        # Handle data type inconsistencies in numeric columns
        for col in df.columns:
            if df[col].dtype == "object":
                # Try to convert to numeric type
                try:
                    numeric_col = pd.to_numeric(df[col], errors="coerce")
                    if not numeric_col.isna().all():
                        df[col] = numeric_col
                except:
                    pass

        return df
    except Exception as e:
        logger.warning(f"Type harmonization failed: {e}")
        return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame data types to save memory"""
    try:
        for col in df.columns:
            if df[col].dtype == "object":
                continue

            if df[col].dtype in ["int64"]:
                col_min, col_max = df[col].min(), df[col].max()
                if pd.isna(col_min) or pd.isna(col_max):
                    continue

                if col_min >= 0:
                    if col_max < 255:
                        df[col] = df[col].astype("uint8")
                    elif col_max < 65535:
                        df[col] = df[col].astype("uint16")
                    elif col_max < 4294967295:
                        df[col] = df[col].astype("uint32")
                else:
                    if col_min >= -128 and col_max < 127:
                        df[col] = df[col].astype("int8")
                    elif col_min >= -32768 and col_max < 32767:
                        df[col] = df[col].astype("int16")
                    elif col_min >= -2147483648 and col_max < 2147483647:
                        df[col] = df[col].astype("int32")

            elif df[col].dtype in ["float64"]:
                # Check if can safely convert to float32
                if (
                    df[col].min() >= np.finfo(np.float32).min
                    and df[col].max() <= np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype("float32")
    except Exception as e:
        logger.warning(f"Data type optimization failed: {e}")

    return df


def analyze_zarr_structure(zarr_path: str) -> Dict[str, Any]:
    """Analyze Zarr data structure"""
    zarr_root = zarr.open(zarr_path, mode="r")

    structure = {
        "meta_keys": [],
        "data_keys": [],
        "total_size": 0,
        "data_shapes": {},
        "data_dtypes": {},
        "chunk_strategies": [],
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
    num_workers: int = None,
) -> List[Dict[str, Any]]:
    """Create optimized data chunk splitting strategy"""

    if num_workers is None:
        num_workers = int(ray.cluster_resources().get("CPU", 1))

    chunks = []
    chunk_id = 0

    for data_path in zarr_structure["data_keys"]:
        shape = zarr_structure["data_shapes"][data_path]
        dtype = zarr_structure["data_dtypes"][data_path]

        if len(shape) == 0:
            continue

        # Calculate bytes per element
        try:
            element_size = np.dtype(dtype).itemsize
        except:
            element_size = 8  # Default assumption: float64

        # Calculate bytes per row
        row_size = element_size * np.prod(shape[1:]) if len(shape) > 1 else element_size

        # Calculate maximum rows per chunk - more conservative memory usage
        target_chunk_bytes = target_chunk_size_mb * 1024 * 1024
        max_rows_per_chunk = max(1, target_chunk_bytes // row_size)  # Remove *2 buffer

        # Ensure chunk count is not too high
        total_rows = shape[0]
        min_rows_per_chunk = max(
            1, total_rows // (num_workers * 2)
        )  # Changed from 8 to 2, reduce chunk count
        max_rows_per_chunk = max(min_rows_per_chunk, max_rows_per_chunk)

        # Create data chunks
        for start_idx in range(0, total_rows, max_rows_per_chunk):
            end_idx = min(start_idx + max_rows_per_chunk, total_rows)

            chunk_info = {
                "chunk_id": chunk_id,
                "data_path": data_path,
                "slice_info": slice(start_idx, end_idx),
                "estimated_rows": end_idx - start_idx,
                "estimated_size_mb": (end_idx - start_idx) * row_size / 1024 / 1024,
                "meta_path": None,
            }

            # Find corresponding metadata path
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
    """Memory monitoring context manager"""
    try:
        import psutil

        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        yield mem_before
        mem_after = process.memory_info().rss / 1024 / 1024
        logger.info(
            f"Memory usage: {mem_before:.1f}MB -> {mem_after:.1f}MB (increase: {mem_after-mem_before:.1f}MB)"
        )
    except ImportError:
        yield 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert Zarr to Parquet using Ray Data"
    )
    parser.add_argument("zarr_path", help="Path to input Zarr file/directory")
    parser.add_argument(
        "--output-dir", help="Output directory (default: same as input)"
    )
    parser.add_argument("--num-workers", type=int, help="Number of Ray workers")
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=2048,  # Changed from 32 to 1024
        help="Target chunk size in MB",
    )
    parser.add_argument(
        "--block-size-mb",
        type=int,
        default=2048,  # Changed from 128 to 2048
        help="Ray Data block size in MB",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=["snappy", "gzip", "brotli", "lz4", "zstd"],
        help="Parquet compression algorithm",
    )
    parser.add_argument("--ray-address", help="Ray cluster address")
    parser.add_argument(
        "--parallelism", type=int, help="Ray Data parallelism override", default=None
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue processing on errors",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Maximum retries per chunk"
    )

    args = parser.parse_args()

    # Set output directory
    zarr_path = Path(args.zarr_path)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = zarr_path.parent / f"{zarr_path.stem}_parquet"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Ray
    ray_init_kwargs = {"ignore_reinit_error": True}
    if args.ray_address:
        ray_init_kwargs["address"] = args.ray_address
    if args.num_workers:
        ray_init_kwargs["num_cpus"] = args.num_workers

    ray.init(**ray_init_kwargs)

    try:
        with memory_monitor() as initial_memory:
            logger.info(f"Starting Zarr structure analysis: {zarr_path}")
            zarr_structure = analyze_zarr_structure(str(zarr_path))

            logger.info(f"Found data keys: {zarr_structure['data_keys']}")
            logger.info(f"Found metadata keys: {zarr_structure['meta_keys']}")
            logger.info(
                f"Total data size: {zarr_structure['total_size'] / 1024 / 1024:.2f} MB"
            )

            # Create chunking strategy
            chunks = create_chunk_strategy(
                zarr_structure,
                target_chunk_size_mb=args.chunk_size_mb,
                num_workers=args.num_workers
                or int(ray.cluster_resources().get("CPU", 1)),
            )

            logger.info(f"Created {len(chunks)} data chunks")
            total_estimated_size = sum(chunk["estimated_size_mb"] for chunk in chunks)
            logger.info(
                f"Estimated total processing size: {total_estimated_size:.2f} MB"
            )

            # Create Zarr data iterator
            logger.info("Creating Zarr data iterator...")
            chunk_iterator = ZarrChunkIterator(
                str(zarr_path),
                chunks,
                continue_on_error=args.continue_on_error,
                max_retries=args.max_retries,
            )

            # Create Ray Dataset
            logger.info("Creating Ray Dataset...")
            dataset = create_zarr_dataset_from_chunks(chunk_iterator)
            dataset = dataset.prefetch(num_blocks=4)  # Add prefetch

            # Set Ray Data configuration
            ctx = ray.data.DataContext.get_current()
            ctx.execution_options.preserve_order = True
            if args.block_size_mb:
                ctx.target_max_block_size = args.block_size_mb * 1024 * 1024

            # Set parallelism
            if args.parallelism:
                dataset = dataset.repartition(args.parallelism)
            else:
                # Automatically set reasonable parallelism
                num_cpus = int(ray.cluster_resources().get("CPU", 1))
                optimal_parallelism = min(len(chunks), num_cpus * 4)
                dataset = dataset.repartition(optimal_parallelism)

            logger.info(f"Dataset parallelism: {dataset.num_blocks()}")

            # Apply data transformation
            logger.info("Starting data transformation...")
            start_time = time.time()

            # Use smaller batch size to avoid memory issues
            processed_dataset = dataset.map_batches(
                prepare_data_batch,
                batch_format="pandas",
                batch_size=8,  # Changed from 1 to 8, process multiple chunks at once
                num_cpus=2,  # Changed from 1 to 2, each task uses more CPUs
                zero_copy_batch=True,
            )

            # Filter empty data frames
            def filter_empty_dataframes(batch):
                """Filter empty data frames"""
                if isinstance(batch, pd.DataFrame) and len(batch) > 0:
                    return batch
                else:
                    return pd.DataFrame()

            final_dataset = processed_dataset.map_batches(
                filter_empty_dataframes,
                batch_format="pandas",
                batch_size=None,
                zero_copy_batch=True,
            )

            # Write Parquet files
            logger.info(f"Writing Parquet files to: {output_dir}")

            # Ray Data's write_parquet method
            final_dataset.write_parquet(
                str(output_dir),
                compression=args.compression,
                try_create_dir=True,
                row_group_size=100000,  # Changed from 25000 to 100000, increase row group size
                use_threads=True,
                num_rows_per_file=1000000,  # Add this parameter, increase rows per file
            )

            conversion_time = time.time() - start_time
            logger.info(
                f"Conversion complete! Time taken: {conversion_time:.2f} seconds"
            )

            # Record error statistics
            if hasattr(chunk_iterator, "error_log") and chunk_iterator.error_log:
                logger.warning(
                    f"Encountered {len(chunk_iterator.error_log)} errors during processing"
                )
                error_file = output_dir / "error_log.json"
                with open(error_file, "w") as f:
                    json.dump(chunk_iterator.error_log, f, indent=2)
                logger.info(f"Error log saved to: {error_file}")

            # Create dataset metadata
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
                    "avg_chunk_size_mb": (
                        total_estimated_size / len(chunks) if chunks else 0
                    ),
                    "total_estimated_size_mb": total_estimated_size,
                },
                "error_count": (
                    len(chunk_iterator.error_log)
                    if hasattr(chunk_iterator, "error_log")
                    else 0
                ),
                "continue_on_error": args.continue_on_error,
                "max_retries": args.max_retries,
            }

            # Save metadata
            metadata_file = output_dir / "dataset_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(dataset_metadata, f, indent=2)

            # Validate output
            parquet_files = list(output_dir.glob("*.parquet"))
            total_size = sum(f.stat().st_size for f in parquet_files)

            logger.info("=" * 60)
            logger.info("Conversion completion statistics:")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Generated files: {len(parquet_files)}")
            logger.info(f"Total output size: {total_size / 1024 / 1024:.2f} MB")
            if zarr_structure["total_size"] > 0:
                logger.info(
                    f"Compression ratio: {zarr_structure['total_size'] / total_size:.2f}x"
                )
            logger.info(
                f"Conversion speed: {zarr_structure['total_size'] / 1024 / 1024 / conversion_time:.2f} MB/s"
            )
            logger.info(f"Metadata file: {metadata_file}")
            if hasattr(chunk_iterator, "error_log") and chunk_iterator.error_log:
                logger.info(
                    f"Error count: {len(chunk_iterator.error_log)} (see error_log.json)"
                )
            logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during conversion process: {e}")
        logger.error(f"Detailed error information: {traceback.format_exc()}")
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

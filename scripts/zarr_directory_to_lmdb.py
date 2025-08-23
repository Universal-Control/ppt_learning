import zarr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "source_path",
    help="Path to source Zarr directory store"
)
parser.add_argument(
    "target_path",
    help="Path to target LMDB store directory"
)
args = parser.parse_args()

store1 = zarr.DirectoryStore(args.source_path)
store2 = zarr.LMDBStore(args.target_path)
zarr.copy_store(store1, store2)
import os
import torch
import argparse
import copy

from ppt_learning.utils.ranging_depth_utils import get_model
from ppt_learning.paths import PPT_DIR

WIDTH, HEIGHT = 672, 504

def export_model_as_jit(model, path, example_color, example_depth):
    os.makedirs(path, exist_ok=True)

    traced_script_module = torch.jit.script(model.inferece, [example_color, example_depth])
    traced_script_module.save(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    save_path = '/'.join(args.model_path.split('/')[:-1]+[args.model_path.split('/')[-1]+'_jit'])

    if 'WORKSPACE' not in os.environ:
        os.environ['WORKSPACE'] = f'{PPT_DIR}/third_party/ranging_depth'
    depth_model = get_model(args.model_path).to(device)

    example_depth = torch.randn(1, 1, WIDTH, HEIGHT).to(device)
    example_color = torch.randn(1, 3, WIDTH, HEIGHT).to(device)
    
    export_model_as_jit(depth_model, save_path, example_color. example_depth)
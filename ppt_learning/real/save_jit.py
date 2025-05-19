import os,sys
import torch
import argparse
import copy
import time

from torch2trt import torch2trt
import torch_tensorrt

from ppt_learning.paths import PPT_DIR
from ppt_learning.utils.ranging_depth_utils import get_model

WIDTH, HEIGHT = 672, 504

EPS = 1e-3

class JitModelExporter(torch.nn.Module):
    def __init__(self, depth_model):
        super().__init__()
        for para in depth_model.parameters():
            para.requires_grad = False
        self.depth_model = depth_model

        class WarpMinMax:
            def warp(self, depth, reference, **kwargs):
                depth_min, depth_max = reference.reshape(depth.shape[0], -1).min(1, keepdim=True)[0], reference.reshape(depth.shape[0], -1).max(1, keepdim=True)[0]
                depth_max[(depth_max - depth_min)<EPS] = depth_min[(depth_max - depth_min)<EPS] + EPS
                return (depth - depth_min[:, None, None]) / (depth_max - depth_min)[:, None, None]
            def unwarp(self, depth, reference, **kwargs):
                depth_min, depth_max = reference.reshape(depth.shape[0], -1).min(1, keepdim=True)[0], reference.reshape(depth.shape[0], -1).max(1, keepdim=True)[0]
                depth_max[(depth_max - depth_min)<EPS] = depth_min[(depth_max - depth_min)<EPS] + EPS
                return depth * (depth_max - depth_min)[:, None, None] + depth_min[:, None, None]
        
        self.warp_func = WarpMinMax()

    def forward(self, x, lowres_depth):
        print(x.dtype, lowres_depth.dtype, self.depth_model._mean.dtype, self.depth_model._std.dtype)
        depth = self.depth_model((x - self.depth_model._mean) / self.depth_model._std, lowres_depth=self.warp_func.warp(lowres_depth, reference=lowres_depth)).unsqueeze(1)
        return self.warp_func.unwarp(depth, reference=lowres_depth) 

    # def forward(self, x, lowres_depth):
    #     depth = self.depth_model((x - self.depth_model._mean) / self.depth_model._std, lowres_depth=lowres_depth).unsqueeze(1)
    #     return depth

def export_model_as_jit(model, path, example_color, example_depth):
    traced_module = torch.jit.trace(JitModelExporter(model), [example_color, example_depth])
    traced_module.save(path)

    return traced_module

def export_model_as_trt(model, path, example_color, example_depth):
    # torch_tensorrt.logging.set_level("DEBUG")
    trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs=[example_color, example_depth], enabled_precisions={torch.float32})
    torch_tensorrt.save(trt_gm, "trt.ts", output_format="torchscript", inputs=[example_color, example_depth])

    # trt_gm = torch2trt(model, [example_color, example_depth], verbose=True)
    # trt_gm.save(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='~/480pnoise-e096-s397312.ckpt')
    parser.add_argument('--num_runs', type=int, default=10)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    jit_save_path = '/'.join(args.model_path.split('/')[:-1]+[args.model_path.split('/')[-1]+'.jit'])
    trt_save_path = '/'.join(args.model_path.split('/')[:-1]+[args.model_path.split('/')[-1]+'.trt'])

    example_depth = torch.randn(1, 1, WIDTH, HEIGHT, dtype=torch.float32).to(device)
    example_color = torch.ones(1, 3, WIDTH, HEIGHT, dtype=torch.float32).to(device)

    if 'RA_WORKSPACE' not in os.environ:
        os.environ['RA_WORKSPACE'] = f'{PPT_DIR}/third_party/ranging_depth'
    depth_model = get_model(args.model_path).to(device)
    
    traced_module = export_model_as_jit(depth_model, jit_save_path, example_color, example_depth)
    import ipdb; ipdb.set_trace()
    print(model.graph)
    model_trt = export_model_as_trt(traced_module, trt_save_path, example_color, example_depth)

    # jit_model = torch.jit.load(save_path).to(device)

    # # Warm-up runs
    # for _ in range(5):
    #     _ = jit_model(example_color, example_depth)
    #     _ = depth_model.inference(example_color, example_depth)

    # jit_total_time = 0
    # raw_total_time = 0

    # for _ in range(args.num_runs):
    #     example_depth = torch.randn(1, 1, WIDTH, HEIGHT, dtype=torch.float32).to(device)
    #     example_color = torch.ones(1, 3, WIDTH, HEIGHT, dtype=torch.float32).to(device)
        
    #     torch.cuda.synchronize() if device == "cuda" else None
    #     time1 = time.time()
    #     res1 = jit_model(example_color, example_depth)
    #     torch.cuda.synchronize() if device == "cuda" else None
    #     jit_total_time += time.time() - time1
    #     print("jit:", time.time() - time1)

    #     example_depth = torch.randn(1, 1, WIDTH, HEIGHT, dtype=torch.float32).to(device)
    #     example_color = torch.ones(1, 3, WIDTH, HEIGHT, dtype=torch.float32).to(device)
        
    #     torch.cuda.synchronize() if device == "cuda" else None
    #     time2 = time.time()
    #     res2 = depth_model.inference(example_color, example_depth)
    #     torch.cuda.synchronize() if device == "cuda" else None
    #     raw_total_time += time.time() - time2
    #     print("raw:", time.time() - time2)

    # print(f"JIT model average time: {jit_total_time / args.num_runs}")
    # print(f"Raw model average time: {raw_total_time / args.num_runs}")
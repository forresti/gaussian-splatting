import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from copy import deepcopy

# conda activate gaussian_splatting
# cd C:\Users\fni\code\gs
# python edit_ply.py -m .\models\truck -s .\data\tandt_db\tandt\truck
# python edit_ply.py -m .\\output\36a65963-9 -s .\data\ficus

def masked_select_multidim(input, mask):
    assert(len(input.shape) == 2)
    x,y = input.shape
    
    outputs = []
    for i in range(y):
        outputs.append(torch.masked_select(input[:,i], mask))
    
    return torch.stack(outputs, dim=-1)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    iteration = -1
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    gaussians2 = deepcopy(gaussians)

    scaling = gaussians._scaling

    # gt stands for greater-than, not ground-truth
    scaling_gt = torch.abs(scaling).gt(8)

    # choose gaussians that are big in all 3 dimensions
    scaling_gt_1d = torch.bitwise_and( torch.bitwise_and(scaling_gt[:,0], scaling_gt[:,1]), scaling_gt[:,2]) 

    gaussians2._scaling = masked_select_multidim(gaussians._scaling, scaling_gt_1d)
    gaussians2._xyz = masked_select_multidim(gaussians._xyz, scaling_gt_1d)

    features_dc = gaussians._features_dc.squeeze(1) # [#gaussians, 1, 3] -> [#gaussians, 3]
    features_dc = masked_select_multidim(features_dc, scaling_gt_1d)
    gaussians2._features_dc = features_dc.unsqueeze(-2) # [#gaussians, 3] -> [#gaussians, 1, 3]
 

    features_rest_list = []
    x,y,z = gaussians._features_dc.shape
    for i in range(z): 
        features_rest_list.append(masked_select_multidim(gaussians._features_rest[:,:,i], scaling_gt_1d))
    gaussians2._features_rest = torch.stack(features_rest_list, dim=-1)

    """
    TODO:
    (Pdb) gaussians._features_dc.shape
    torch.Size([3405153, 1, 3])
    (Pdb) gaussians._features_rest.shape
    torch.Size([3405153, 15, 3])
    """
    gaussians2._opacity = masked_select_multidim(gaussians._opacity, scaling_gt_1d)
    gaussians2._scaling = masked_select_multidim(gaussians._scaling, scaling_gt_1d)
    gaussians2._rotation = masked_select_multidim(gaussians._rotation, scaling_gt_1d)

    # import pdb; pdb.set_trace()

    savedir = "./fni_outputs"
    os.makedirs(savedir, exist_ok=True)
    gaussians2.save_ply(f"{savedir}/point_cloud.ply")



    

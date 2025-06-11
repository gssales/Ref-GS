#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_real
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, render_normals, ITER=0, ENV_CENTER=None, ENV_RADIUS=None, XYZ=[0,1,2]):
    render_path = os.path.join(model_path, name, "ref_gs_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ref_gs_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if render_normals:
        normals_path = os.path.join(model_path, name, "ref_gs_{}".format(iteration), "normals")
        makedirs(normals_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render_real(view, gaussians, pipeline, background, iteration=iteration, ITER=ITER, ENV_CENTER=ENV_CENTER, ENV_RADIUS=ENV_RADIUS, XYZ=XYZ) 
        rendering = render_pkg["pbr_rgb"]
        gt = view.original_image

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        if render_normals:
            normals = render_pkg["rend_normal"]
            normals = normals*0.5+0.5
            torchvision.utils.save_image(normals, os.path.join(normals_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, render_normals : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        ITER = dataset.init_until_iter
        ENV_CENTER = torch.tensor([float(c) for c in dataset.env_scope_center], device='cuda')
        ENV_RADIUS = dataset.env_scope_radius
        XYZ = [int(float(c)) for c in dataset.xyz_axis]

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_normals, ITER=ITER, ENV_CENTER=ENV_CENTER, ENV_RADIUS=ENV_RADIUS, XYZ=XYZ)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, render_normals, ITER=ITER, ENV_CENTER=ENV_CENTER, ENV_RADIUS=ENV_RADIUS, XYZ=XYZ)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--render_normals", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.render_normals)
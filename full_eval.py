import os
from argparse import ArgumentParser
import time

ref_real_scenes = ["ref_real/sedan", "ref_real/gardenspheres", "ref_real/toycar"]
refnerf_scenes = ["shiny_blender/helmet","shiny_blender/car","shiny_blender/ball","shiny_blender/teapot","shiny_blender/coffee","shiny_blender/toaster"]
nerf_synthetic_scenes = ["nerf_synthetic/ship","nerf_synthetic/ficus","nerf_synthetic/lego","nerf_synthetic/mic","nerf_synthetic/hotdog","nerf_synthetic/chair","nerf_synthetic/materials","nerf_synthetic/drums"]
glossy_synthetic_scenes = ["GlossySynthetic/bell","GlossySynthetic/tbell","GlossySynthetic/potion","GlossySynthetic/teapot","GlossySynthetic/luyu","GlossySynthetic/cat"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="/mnt/output/ref_gs/eval")

extra_args = {
    "ref_real/sedan": " -r 8 --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --env_scope_center -0.032 0.808 0.751 --env_scope_radius 2.138 --init_until_iter 700 --xyz_axis 2.0 1.0 0.0",
    "ref_real/gardenspheres": " -r 6 --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --env_scope_center -0.2270 1.9700 1.7740 --env_scope_radius 0.974 --init_until_iter 700 --xyz_axis 2.0 1.0 0.0",
    "ref_real/toycar": " -r 6 --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --env_scope_center 0.486 1.108 3.72 --env_scope_radius 2.507 --init_until_iter 1500 --xyz_axis 0.0 2.0 1.0",
    
    "shiny_blender/helmet": " --run_dim 256 --albedo_bias 0",
    "shiny_blender/car": " --run_dim 256 --albedo_bias 0",
    "shiny_blender/ball": " --run_dim 256 --albedo_bias 0",
    "shiny_blender/teapot": " --run_dim 256 --albedo_bias 0",
    "shiny_blender/coffee": " --run_dim 256 --albedo_bias 0 --albedo_lr 0.002",
    "shiny_blender/toaster": " --run_dim 256 --albedo_bias 0",

    "nerf_synthetic/ship": " --run_dim 64 --albedo_bias 0 --gsrgb_loss --albedo_lr 0.002",
    "nerf_synthetic/ficus": " --run_dim 64 --albedo_bias 0 --gsrgb_loss --albedo_lr 0.002",
    "nerf_synthetic/lego": " --run_dim 64 --albedo_bias 0 --gsrgb_loss --albedo_lr 0.002",
    "nerf_synthetic/mic": " --run_dim 64 --albedo_bias 0 --gsrgb_loss --albedo_lr 0.002",
    "nerf_synthetic/hotdog": " --run_dim 64 --albedo_bias 0 --gsrgb_loss --albedo_lr 0.002",
    "nerf_synthetic/chair": " --run_dim 64 --albedo_bias 0 --gsrgb_loss --albedo_lr 0.002",
    "nerf_synthetic/materials": " --run_dim 256 --albedo_bias 0",
    "nerf_synthetic/drums": " --run_dim 64 --albedo_bias 0 --albedo_lr 0.002",

    "GlossySynthetic/bell": " --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --init_until_iter 3000",
    "GlossySynthetic/tbell": " --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --init_until_iter 3000",
    "GlossySynthetic/potion": " --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --init_until_iter 3000",
    "GlossySynthetic/teapot": " --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --init_until_iter 3000",
    "GlossySynthetic/luyu": " --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --init_until_iter 3000",
    "GlossySynthetic/cat": " --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --init_until_iter 3000",
}


args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(ref_real_scenes)
all_scenes.extend(refnerf_scenes)
all_scenes.extend(nerf_synthetic_scenes)
all_scenes.extend(glossy_synthetic_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--ref_real', type=str, default="/mnt/data/ref_real")
    parser.add_argument('--refnerf', type=str, default="/mnt/data/shiny_blender")
    parser.add_argument('--nerf_synthetic', type=str, default="/mnt/data/nerf_synthetic")
    parser.add_argument('--glossy_synthetic', type=str, default="/mnt/data/GlossySynthetic")
    args = parser.parse_args()
if not args.skip_training:
    common_args = " --disable_viewer --quiet --eval --test_iterations -1 --save_iterations 7000 30000"

    start_time = time.time()
    for scene in ref_real_scenes:
        source = args.ref_real + "/" + scene
        extra = extra_args[scene]
        os.system("python train-real.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + extra)
    ref_real_timing = (time.time() - start_time)/60.0
    
    start_time = time.time()
    for scene in refnerf_scenes:
        source = args.refnerf + "/" + scene
        extra = extra_args[scene]
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + extra)
    refnerf_timing = (time.time() - start_time)/60.0
    
    start_time = time.time()
    for scene in nerf_synthetic_scenes:
        source = args.nerf_synthetic + "/" + scene
        extra = extra_args[scene]
        os.system("python train-NeRF.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + extra)
    nerf_synthetic_timing = (time.time() - start_time)/60.0
    
    start_time = time.time()
    for scene in glossy_synthetic_scenes:
        source = args.glossy_synthetic + "/" + scene
        extra = extra_args[scene]
        os.system("python train-NeRO.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + extra)
    glossy_synthetic_timing = (time.time() - start_time)/60.0

with open(os.path.join(args.output_path,"timing.txt"), 'w') as file:
    file.write(f"ref_real: {ref_real_timing} minutes \n shiny_blender: {refnerf_timing} minutes \n nerf_synthetic: {nerf_synthetic_timing} minutes \n GlossySynthetic: {glossy_synthetic_timing} minutes \n")

if not args.skip_rendering:
    all_sources = []
    for scene in ref_real_scenes:
        all_sources.append(args.ref_real + "/" + scene)
    for scene in refnerf_scenes:
        all_sources.append(args.refnerf + "/" + scene + " --render_normals")
    for scene in nerf_synthetic_scenes:
        all_sources.append(args.nerf_synthetic + "/" + scene)
    for scene in glossy_synthetic_scenes:
        all_sources.append(args.glossy_synthetic + "/" + scene)
    
    common_args = " --quiet --eval --skip_train"

    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("python metrics.py -m " + scenes_string)

import os
from argparse import ArgumentParser
import time

ref_real_scenes = ["sedan", "gardenspheres", "toycar"]
refnerf_scenes = ["helmet","car","ball","teapot","coffee","toaster"]
nerf_synthetic_scenes = ["ship","ficus","lego","mic","hotdog","chair","materials","drums"]
glossy_synthetic_scenes = ["bell","tbell","potion","teapot","luyu","cat"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="/mnt/output/ref_gs/eval")

extra_args = {
    "sedan": " -r 8 --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --env_scope_center -0.032 0.808 0.751 --env_scope_radius 2.138 --init_until_iter 700 --xyz_axis 2.0 1.0 0.0",
    "gardenspheres": " -r 6 --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --env_scope_center -0.2270 1.9700 1.7740 --env_scope_radius 0.974 --init_until_iter 700 --xyz_axis 2.0 1.0 0.0",
    "toycar": " -r 6 --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --env_scope_center 0.486 1.108 3.72 --env_scope_radius 2.507 --init_until_iter 1500 --xyz_axis 0.0 2.0 1.0",
    
    "helmet": " --run_dim 256 --albedo_bias 0",
    "car": " --run_dim 256 --albedo_bias 0",
    "ball": " --run_dim 256 --albedo_bias 0",
    "teapot": " --run_dim 256 --albedo_bias 0",
    "coffee": " --run_dim 256 --albedo_bias 0 --albedo_lr 0.002",
    "toaster": " --run_dim 256 --albedo_bias 0",

    "ship": " --run_dim 64 --albedo_bias 0 --gsrgb_loss --albedo_lr 0.002",
    "ficus": " --run_dim 64 --albedo_bias 0 --gsrgb_loss --albedo_lr 0.002",
    "lego": " --run_dim 64 --albedo_bias 0 --gsrgb_loss --albedo_lr 0.002",
    "mic": " --run_dim 64 --albedo_bias 0 --gsrgb_loss --albedo_lr 0.002",
    "hotdog": " --run_dim 64 --albedo_bias 0 --gsrgb_loss --albedo_lr 0.002",
    "chair": " --run_dim 64 --albedo_bias 0 --gsrgb_loss --albedo_lr 0.002",
    "materials": " --run_dim 256 --albedo_bias 0",
    "drums": " --run_dim 64 --albedo_bias 0 --albedo_lr 0.002",

    "bell": " --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --init_until_iter 3000",
    "tbell": " --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --init_until_iter 3000",
    "potion": " --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --init_until_iter 3000",
    "teapot": " --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --init_until_iter 3000",
    "luyu": " --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --init_until_iter 3000",
    "cat": " --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --init_until_iter 3000",
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
        os.system("python train-real.py -s " + source + " -m " + args.output_path + "/ref_real/" + scene + common_args + extra)
    ref_real_timing = (time.time() - start_time)/60.0
    
    start_time = time.time()
    for scene in refnerf_scenes:
        source = args.refnerf + "/" + scene
        extra = extra_args[scene]
        os.system("python train.py -s " + source + " -m " + args.output_path + "/shiny_blender/" + scene + common_args + extra)
    refnerf_timing = (time.time() - start_time)/60.0
    
    start_time = time.time()
    for scene in nerf_synthetic_scenes:
        source = args.nerf_synthetic + "/" + scene
        extra = extra_args[scene]
        os.system("python train-NeRF.py -s " + source + " -m " + args.output_path + "/nerf_synthetic/" + scene + common_args + extra)
    nerf_synthetic_timing = (time.time() - start_time)/60.0
    
    start_time = time.time()
    for scene in glossy_synthetic_scenes:
        source = args.glossy_synthetic + "/" + scene
        extra = extra_args[scene]
        os.system("python train-NeRO.py -s " + source + " -m " + args.output_path + "/GlossySynthetic/" + scene + common_args + extra)
    glossy_synthetic_timing = (time.time() - start_time)/60.0

with open(os.path.join(args.output_path,"timing.txt"), 'w') as file:
    file.write(f"ref_real: {ref_real_timing} minutes \n shiny_blender: {refnerf_timing} minutes \n nerf_synthetic: {nerf_synthetic_timing} minutes \n GlossySynthetic: {glossy_synthetic_timing} minutes \n")

if not args.skip_rendering:
    all_sources = []
    for scene in ref_real_scenes:
        all_sources.append(args.ref_real + "/" + scene)
    for scene in refnerf_scenes:
        all_sources.append(args.refnerf + "/" + scene)
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

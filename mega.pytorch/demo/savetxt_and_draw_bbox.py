import glob
import os
import argparse

from mega_core.config import cfg
from predictor import VIDDemo

#use which gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description="PyTorch Object Detection Visualization")
parser.add_argument(
    "--method",
    choices=["base", "dff", "fgfa", "rdn", "mega"],
    default="fgfa",
    type=str,
    help="which method to use",
)
parser.add_argument(
    "--config",
    default="/usr/idip/idip/liuan/project/VOD-project/MEGA/mega.pytorch/configs/FGFA/vid_R_50_C4_FGFA_1x.yaml",
    metavar="FILE",
    help="path to config file",
)
parser.add_argument(
    "--checkpoint",
    default="/usr/idip/idip/liuan/project/VOD-project/MEGA/mega.pytorch/training_dir/FGFA_R_50_C4_1x_1GPU/model_final.pth",
    help="The path to the checkpoint for test.",
    type=str,
)
parser.add_argument(
    "--suffix",
    # default=".JPEG",
    default=".jpg",
    help="the suffix of the images in the image folder.",
)
parser.add_argument(
    "--test_img_folder",
    # default="demo/visualization/base",
    # default="/usr/idip/idip/liuan/data/make_Object_Detect_format_dataset/4class_test/jpg/",
    default="/usr/idip/idip/liuan/data/make_Object_Detect_format_dataset/problem_case/problem_jpg/",
    help="where to test image folder.",
)
parser.add_argument(
    "--output_folder",
    # default="demo/visualization/base",
    # default="/usr/idip/idip/liuan/data/save_txt/VID_result/AP50/DFF_R_50_C4_1x_1GPU_final_model/draw_bbox/",
    default="/usr/idip/idip/liuan/data/save_txt/VID_result/AP50/FGFA_R_50_C4_1x_1GPU_final_model/draw_bbox",
    help="where to store the visulization result.",
)
parser.add_argument(
    "--savetxt_folder",
    # default="demo/visualization/base",
    # default="/usr/idip/idip/liuan/data/save_txt/VID_result/AP50/DFF_R_50_C4_1x_1GPU_final_model/txt/",
    default="/usr/idip/idip/liuan/data/save_txt/VID_result/AP50/FGFA_R_50_C4_1x_1GPU_final_model/txt",
    help="where to store the txt result.",
)
parser.add_argument(
    "--video",
    action="store_true", # action是开关的作用，默认是False，当python demo.py --video是True
    help="if True, input a video for visualization.",
)

args = parser.parse_args()
cfg.merge_from_file("/usr/idip/idip/liuan/project/VOD-project/MEGA/mega.pytorch/configs/BASE_RCNN_1gpu.yaml")
cfg.merge_from_file(args.config)
cfg.merge_from_list(["MODEL.WEIGHT", args.checkpoint])

vid_demo = VIDDemo(
    cfg,
    savetxt_path = args.savetxt_folder,
    method=args.method,
    confidence_threshold=0.7,
    output_folder=args.output_folder
)

if not args.video:
    vid_demo.run_on_image_folder_savetxt_drawbbox(args.test_img_folder, suffix=args.suffix)
else:
    print("args.video has something wrong !")


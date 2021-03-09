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
    default="dff",
    type=str,
    help="which method to use",
)
parser.add_argument(
    "--config",
    default="/usr/idip/idip/liuan/project/VOD-project/MEGA/mega.pytorch/configs/DFF/vid_R_50_C4_DFF_1x.yaml",
    metavar="FILE",
    help="path to config file",
)
parser.add_argument(
    "--checkpoint",
    default="/usr/idip/idip/liuan/project/VOD-project/MEGA/mega.pytorch/training_dir/DFF_R_50_C4_1x_1GPU/model_final.pth",
    help="The path to the checkpoint for test.",
    type=str,
)
parser.add_argument(
    "--suffix",
    default=".JPEG",
    # default=".jpg",
    help="the suffix of the images in the image folder.",
)
parser.add_argument(
    "--test_img_folder",
    # default="demo/visualization/base",
    # default="/usr/idip/idip/liuan/data/make_Object_Detect_format_dataset/4class_test/jpg/",
    default="/usr/idip/idip/liuan/project/VOD-project/MEGA/mega.pytorch/datasets/ILSVRC2015/Data/VID/test/",
    help="where to test image folder.",
)
parser.add_argument(
    "--output_folder",
    # default="demo/visualization/base",
    # default="/usr/idip/idip/liuan/data/save_txt/VID_result/AP50/DFF_R_50_C4_1x_1GPU_final_model/draw_bbox/",
    default="/usr/idip/idip/liuan/data/save_txt/VID_result/test_res/DFF_resnet50",
    help="where to store the visulization result.",
)
# parser.add_argument(
#     "--savetxt_folder",
#     # default="demo/visualization/base",
#     # default="/usr/idip/idip/liuan/data/save_txt/VID_result/AP50/DFF_R_50_C4_1x_1GPU_final_model/txt/",
#     default="/usr/idip/idip/liuan/data/save_txt/VID_result/AP50/FGFA_R_50_C4_1x_1GPU_final_model/txt",
#     help="where to store the txt result.",
# )
parser.add_argument(
    "--video",
    action="store_true", # action是开关的作用，默认是False，当python demo.py --video是True
    help="if True, input a video for visualization.",
)

args = parser.parse_args()
cfg.merge_from_file("/usr/idip/idip/liuan/project/VOD-project/MEGA/mega.pytorch/configs/BASE_RCNN_1gpu.yaml")
cfg.merge_from_file(args.config)
cfg.merge_from_list(["MODEL.WEIGHT", args.checkpoint])

test_video_list = os.listdir(args.test_img_folder) # 测试视频的名称列表
if not os.path.exists(os.path.join(args.output_folder, 'draw_bbox')):
    os.makedirs(os.path.join(args.output_folder, 'draw_bbox'))
if not os.path.exists(os.path.join(args.output_folder, 'txt')):
    os.makedirs(os.path.join(args.output_folder, 'txt'))
for video_name in test_video_list:
    video_savejpg_path = os.path.join(args.output_folder, 'draw_bbox')
    video_savetxt_path = os.path.join(args.output_folder, 'txt')
    cur_test_path = os.path.join(args.test_img_folder, video_name)
    cur_savejpg_path = os.path.join(video_savejpg_path, video_name)
    cur_savetxt_path = os.path.join(video_savetxt_path, video_name)
    if not os.path.exists(cur_savejpg_path):
        os.makedirs(cur_savejpg_path)
    if not os.path.exists(cur_savetxt_path):
        os.makedirs(cur_savetxt_path)

    vid_demo = VIDDemo(
        cfg,
        savetxt_path = cur_savetxt_path,
        method=args.method,
        confidence_threshold=0.7,
        output_folder=cur_savejpg_path
    )

    if not args.video:
        vid_demo.run_on_image_folder_savetxt_drawbbox(cur_test_path, suffix=args.suffix)
    else:
        print("args.video has something wrong !")


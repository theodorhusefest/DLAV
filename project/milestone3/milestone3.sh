video_path=$1 # /home/minoda/koji_hdd/datasets/DLAV_videos/MOT16-10-raw.webm
video_name=$2

path_to_weight_YOLOv3=/home/minoda/yolov3_180520.pth
path_to_weight_ABDNet=/home/minoda/msmt17_final_best.pth.zip.tar

rm -rf outputs/$2

# video -> images
echo "\n######## Generate images from video ########"
python3 ./utils/extract_images_from_video.py ${video_path} outputs/$2/images

# Perform YOLOv3 for pedestrian detection
echo "\n######## Start Object Detection by YOLOv3 ########"
python3 ../PyTorch-YOLOv3/generate_person_images.py \
        --image_folder outputs/$2/images \
        --outputs_folder outputs/$2/yolov3_outputs \
        --model_def ../PyTorch-YOLOv3/config/yolov3-custom.cfg \
        --weights_path ${path_to_weight_YOLOv3} \
        --class_path ../PyTorch-YOLOv3/data/ecp/ecp.names \
        --save_thres 0.99

# Perform re-ID
echo "\n######## Start re-ID by ABD-Net ########"
python3 ../ABD-Net/classify.py -t pedestrianreid -s not_used \
        --load-weights=${path_to_weight_ABDNet} \
        --root=./outputs \
        --video=${video_name} \
        --save-dir=output


# Generate deliberative
echo "\n######## Generate video with detected pedestrians ########"
python3 ./utils/generate_video.py \
        outputs/$2/images outputs/$2/yolov3_outputs/query_list.csv utils/id2rgb.txt ./outputs/$2
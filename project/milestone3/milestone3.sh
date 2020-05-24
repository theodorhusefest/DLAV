video_path=$1 # /home/minoda/koji_hdd/datasets/DLAV_videos/MOT16-10-raw.webm

save_folder=./intermediates/images
pedestrians_folder=./intermediates/yolov3_outputs
path_to_weight_YOLOv3=../PyTorch-YOLOv3/weights/yolov3_300420.pth
path_to_weight_ABDNet=/path/to/ABDNet/weight

rm -rf ${save_folder}
rm -rf ${pedestrians_folder}

# video -> images
python3 ./utils/extract_images_from_video.py ${video_path} ${save_folder}

# Perform YOLOv3 for pedestrian detection
python3 ../PyTorch-YOLOv3/generate_person_images.py \
        --image_folder ${save_folder} \
        --outputs_folder ${pedestrians_folder} \
        --model_def ../PyTorch-YOLOv3/config/yolov3-custom.cfg \
        --weights_path ${path_to_weight_YOLOv3} \
        --class_path ../PyTorch-YOLOv3/data/ecp/ecp.names \
        --save_thres 0.95

# Perform re-ID
# python ABDNet/inference.py

# Generate deliberative
python3 ./utils/generate_video.py \
        ${save_folder} ${pedestrians_folder}/detected_pedestrians.csv ./
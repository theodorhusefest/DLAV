video_path=$1 # /home/minoda/koji_hdd/datasets/DLAV_videos/MOT16-10-raw.webm

export CUDA_VISIBLE_DEVICES=2

thres_yolov3=0.90

save_folder=./intermediates/images
pedestrians_folder=./intermediates/yolov3_outputs
path_to_weight_YOLOv3=../PyTorch-YOLOv3/weights/yolov3_300420.pth
path_to_weight_ABDNet=/path/to/ABDNet/weight

rm -rf ${save_folder}
rm -rf ${pedestrians_folder}

# video -> images
echo 1. Extracting images from the video...
python3 ./utils/extract_images_from_video.py ${video_path} ${save_folder}

# Perform YOLOv3 for pedestrian detection
echo 2. Start detecting pedestrians...
python3 ../PyTorch-YOLOv3/generate_person_images.py \
        --image_folder ${save_folder} \
        --outputs_folder ${pedestrians_folder} \
        --model_def ../PyTorch-YOLOv3/config/yolov3-custom.cfg \
        --weights_path ${path_to_weight_YOLOv3} \
        --class_path ../PyTorch-YOLOv3/data/ecp/ecp.names \
        --save_thres ${thres_yolov3}

# Perform re-ID
echo 3. Start performing re-ID with ABDNet...
echo Not Implemented
# python ABDNet/inference.py

# Generate deliberative
echo 4. Reconstructing video with detected bounding boxes...
python3 ./utils/generate_video.py \
        ${save_folder} ${pedestrians_folder}/detected_pedestrians.csv ./

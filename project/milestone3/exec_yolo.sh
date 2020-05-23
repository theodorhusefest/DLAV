# Change these three variables for your configuration
video_path=$1 # /home/minoda/koji_hdd/datasets/DLAV_videos/MOT16-10-raw.webm
workdir=$2 
weight_path=$3
save_folder=${workdir}/images
out_folder=${workdir}/yolov3_outputs

python3 extract_images_from_video.py ${video_path} ${save_folder}

python3 ../PyTorch-YOLOv3/generate_person_images.py \
        --image_folder ${save_folder} \
        --outputs_folder ${out_folder} \
        --model_def ../PyTorch-YOLOv3/config/yolov3-custom.cfg \
        --weights_path ${weight_path} \
        --class_path ../PyTorch-YOLOv3/data/ecp/ecp.names \
        --save_thres 0.99

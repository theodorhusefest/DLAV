video_path=/home/minoda/hoge.mp4
save_folder=/home/minoda/git/DLAV/project/milestone3/images
out_folder=/home/minoda/git/DLAV/project/milestone3/yolov3_outputs

ffmpeg -i ${video_path} -r 20 ${save_folder}/img_%04d.png

python3 ../PyTorch-YOLOv3/generate_person_images.py \
        --image_folder ${save_folder} \
        --outputs_folder ${out_folder} \
        --model_def ../PyTorch-YOLOv3/config/yolov3-custom.cfg \
        --weights_path ../PyTorch-YOLOv3/weights/yolov3_300420.pth \
        --class_path ../PyTorch-YOLOv3/data/ecp/ecp.names \
        --save_thres 0.99
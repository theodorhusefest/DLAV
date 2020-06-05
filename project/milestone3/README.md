# Description for Milestone 3

### Setup
```
$ git clone --recursive https://github.com/theodorhusefest/DLAV.git  
```
In addition, please follow the instructions for PyTorch-YOLOv3 and ABD-Net and install the requirements.  
```
$ pip install -r /path/to/DLAV/repos/project/PyTorch-YOLOv3/requirements.txt  
$ pip install -r /path/to/DLAV/repos/project/ABD-Net/requirements.txt  
```  
You also need to download pretrained weights for YOLOv3 and ABD-Net from here. Do not forget to edit `path_to_weight_YOLOv3` and `path_to_weight_ABDNet` in `milestone3.sh`.  
- [Weight for YOLOv3](https://drive.google.com/open?id=1ZLjJXnZDyFhItueba25a0prB5CDfHB2E)  
- [Weight for ABDNet](https://drive.google.com/open?id=1WkvBBda2GP0-uJby5qb5dvHBggCOQ6il) 

### How to generate video from original video
Just run the following command.  
```
$ cd /path/to/DLAV/repos/project/milestone3  
$ source milestone3.sh /path/to/video video_name 
```

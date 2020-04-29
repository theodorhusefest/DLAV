# Network Checkpoints

## AWS location (example)

    /home/checkpoints/
        YOLOv3/
            01/
                checkpoint.ckpt
                tensorboard-log-file
            02/
            03/
            ...
         PEDESTRON/
            01/
            02/
            03/
            ...


## YOLOv3

### -- Add your checkpoint name here, e.g. "01" --
* filename 
* hyper parameters used
* data set (splits) used
* epochs on training
* previous checkpoints used
* based on git commit ...
* any issues during training
* good/bad results

### 01: Baseline by Koji@29/04/20
* filename: baseline_290420.pth
* hyper parameters used: default
* data set (splits) used: all train data in ECP/day/
* epochs on training: About 15 epochs
* previous checkpoints used: darknet53.conv.74
* based on git commit: 9a91710
* any issues during training: Nothing
* good/bad results: About 0.07 mAP. Fairly good results for a first run.

## PEDESTRON

### -- Add your checkpoint name here, e.g. "01" --
* hyper parameters used
* data set (splits) used
* previous checkpoints used
* based on git commit ...
* any issues during training
* good/bad results
* ...
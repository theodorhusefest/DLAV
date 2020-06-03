import cv2
import os
from pathlib import Path
import argparse
from natsort import natsorted
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_images", type=str, help="path to images folder")
    parser.add_argument("input_csv", type=str, help="path to csv")
    parser.add_argument("id2rgb", type=str, help="path to id-to-rgb correspondence tsv.")
    parser.add_argument("output", type=str, help="path")
    args = parser.parse_args()
    # print(args)

    # read csv
    df = pd.read_csv(args.input_csv)
    images_list = Path(args.input_images) / "image_%04d.png"

    # VideoCapture 
    cap = cv2.VideoCapture(str(images_list))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30

    # VideoWriter 
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    writer = cv2.VideoWriter(os.path.join(args.output, 'output.mp4'), fourcc, fps, (width, height))
    image_id = 0
    
    # color list
    df_colorlist = pd.read_csv(args.id2rgb, sep='\t', header=None, index_col=0, names=['rgb'])
    df_colorlist.rgb = df_colorlist.rgb.apply(lambda x:[int(val) for val in x[1:-1].split(',')])

    color_list = df_colorlist.rgb.values
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        for box_info in df[df['image_id']==image_id][['x', 'y', 'w', 'h', 'confidence', 'ID']].values:
            [x, y, w, h, conf, ID] = box_info
            x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_list[int(ID%len(color_list))], 3)
            cv2.putText(frame, 'ID:%d' % ID, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2.0, color_list[int(ID%len(color_list))], 2, cv2.LINE_AA)
        writer.write(frame)  
        image_id += 1

    writer.release()
    cap.release()

import cv2
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="specify video")
    parser.add_argument("outputs_folder", type=str, help="output folder for images")
    opt = parser.parse_args()
    print(opt)

    cap = cv2.VideoCapture(opt.video)
    cnt = 0
    os.makedirs(opt.outputs_folder, exist_ok=True)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(opt.outputs_folder, "image_{0:04d}.png".format(cnt)), frame)
        cnt += 1
    cap.release()
import os
import cv2
import argparse
import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--fps', type=int, default=10)
    opt = parser.parse_args()

    img_array = []
    size = (0,0)
    for filename in sorted(glob.glob(os.path.join(opt.input,'*[0-9].*'))):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)


    out = cv2.VideoWriter('test.mp4',
                        cv2.VideoWriter_fourcc(*'DIVX'),
                        opt.fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


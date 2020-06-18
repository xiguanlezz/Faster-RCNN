import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
from configs.config import classes


# classes = ['pedestrian', 'truck', 'car', 'cyclist', 'van', 'tram', 'person_sitting']


def draw_predict(dir_path, img_name, boxes, labels, scores):
    file_path = dir_path + img_name
    im = np.array(Image.open(file_path))
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    for i in range(len(boxes)):
        xmin = boxes[i][0]
        ymin = boxes[i][1]
        width = boxes[i][2] - boxes[i][0]
        height = boxes[i][3] - boxes[i][1]
        text = classes[labels[i]] + ":" + "%d" % (scores[i] * 100) + "%"
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, text, fontsize=8, bbox=dict(facecolor='green', alpha=1))
    savedir = "show_result/"
    if (not os.path.exists(savedir)):
        os.mkdir(savedir)
    plt.savefig(savedir + img_name)
    plt.clf()

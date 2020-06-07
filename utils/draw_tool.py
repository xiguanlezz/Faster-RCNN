import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


def draw_predict(dir_path, img_name, boxes, labels, scores):
    classes = []
    file_path = dir_path + img_name
    im = np.array(Image.open(file_path))
    fig, ax = plt.subplot(1)
    ax.image(im)
    for i in range(len(boxes)):
        xmin = boxes[i][0]
        ymin = boxes[i][1]
        width = boxes[i][2] - boxes[i][0]
        height = boxes[i][3] - boxes[i][1]
        text = classes[labels[i]] + ":" + "%d" % (scores[i] * 100) + "%"
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, text, frontsize=8, bbox=dict(facecolor='green', alpha=1))

    plt.savefig("show_result/" + img_name + ".png")
    plt.clf()

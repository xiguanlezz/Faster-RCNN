import numpy as np


def generate_base_anchors(base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32], center_x=0, center_y=0):
    """
    function description: 生成k个以(0, 0)为中心的anchors模板

    :param base_size: 特征图的每个像素的感受野大小(相当于featuremap上的一个像素的尺度所对应原图上的尺度)
    :param ratios: 高宽的比率
    :param scales: 面积的scales的开方
    :return:
    """
    base_anchor = np.zeros((len(ratios) * len(scales), 4), dtype=np.float32)

    # 生成anchor的算法本质: 使得总面积不变, 一个像素点衍生出9个anchors
    for i in range(len(scales)):
        for j in range(len(ratios)):
            index = i * len(ratios) + j
            area = (base_size * scales[i]) ** 2
            width = np.sqrt(area * 1.0 / ratios[j])
            height = width * ratios[j]

            # 只需要保存左上角个右下角的点的坐标即可
            base_anchor[index, 0] = -width / 2. + center_x
            base_anchor[index, 1] = -height / 2. + center_y
            base_anchor[index, 2] = width / 2. + center_x
            base_anchor[index, 3] = height / 2. + center_y

    return base_anchor


def enumerate_shifted_anchor(base_anchor, base_size, width, height):
    """
    function description: 减少不必要的如generate_base_anchors的计算, 较大的特征图的锚框生成模板, 生成锚框的初选模板即滑动窗口

    :param base_anchor: 需要reshape的anchors
    :param base_size: 特征图的每个像素的感受野大小
    :param height: featuremap的高度
    :param width: featuremap的宽度
    :return:
        anchor: 维度为:[width*height*k, 4]的先验框(anchors)
    """
    # 计算featuremap中每个像素点在原图中感受野上的中心点坐标
    shift_x = np.arange(0, width * base_size, base_size)
    shift_y = np.arange(0, height * base_size, base_size)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    print('shift_x: ', shift_x.shape, 'shift_y: ', shift_y.shape)

    # TODO 感觉最正统的方法还是遍历中心点
    # index = 0
    # for x in shift_x:
    #     for y in shift_y:
    #         anchors = generate_base_anchors(center_x=x, center_y=y)
    #         if index == 0:
    #             old_anchors = anchors
    #         else:
    #             anchors = np.concatenate((old_anchors, anchors), axis=0)
    #             old_anchors = anchors
    #         index += 1

    # TODO 直接利用broadcast貌似也可以达到目的
    # shift_x.ravel()表示原地将为一维数组, shift的维度为: [feature_stride, 4]
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)

    # print("base_anchor: ", base_anchor.shape)
    # print("shift: ", shift.shape)
    # 9个先验框
    A = base_anchor.shape[0]
    K = shift.shape[0]
    anchor = base_anchor.reshape((1, A, 4)) + shift.reshape((K, 1, 4))

    # 最后再合成为所有的先验框, 相当于对featuremap的每个像素点都生成k(9)个先验框(anchors)
    anchors = anchor.reshape((K * A, 4)).astype(np.float32)
    print('result: ', anchors.shape)
    return anchors


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    nine_anchors = generate_base_anchors()
    print(nine_anchors.shape)

    height, width, base_size = 38, 38, 16
    all_anchors = enumerate_shifted_anchor(nine_anchors, base_size, width, height)
    print(38 * 38 * 9)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(-500, 500)
    plt.xlim(-500, 500)
    shift_x = np.arange(0, width * base_size, base_size)
    shift_y = np.arange(0, height * base_size, base_size)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    plt.scatter(shift_x, shift_y)

    box_widths = all_anchors[:, 2] - all_anchors[:, 0]
    box_heights = all_anchors[:, 3] - all_anchors[:, 1]
    print(all_anchors)

    for i in range(18):
        print('width: ', box_widths[i], 'height: ', box_heights[i])
        if i >= 9:
            rect = plt.Rectangle([all_anchors[i, 0], all_anchors[i, 1]], box_widths[i],
                                 box_heights[i], color="r", fill=False)
        else:
            rect = plt.Rectangle([all_anchors[i, 0], all_anchors[i, 1]], box_widths[i],
                                 box_heights[i], color="b", fill=False)
        ax.add_patch(rect)
    plt.show()

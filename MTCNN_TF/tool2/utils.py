import numpy as np


# IOU计算
def iou(box, boxes, isMin = False):  # [x1,y1,x2,y2,c]
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # 加0是因为可能没有事负数，没有意义 就用0
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h

    if isMin:
        ovr = np.true_divide(inter, np.minimum(box_area, area))
    else:
        ovr = np.true_divide(inter, (box_area + area - inter))

    return ovr


# NMS非极大值抑制
def nms(boxes, thresh=0.3, isMin = False):
    if boxes.shape[0] == 0:
        return np.array([])

    _boxes = boxes[(-boxes[:, 4]).argsort()]
    r_boxes = []

    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        b_boxes = _boxes[1:]

        r_boxes.append(a_box)  # 保留最大的

        index = np.where(iou(a_box, b_boxes, isMin) < thresh)
        _boxes = b_boxes[index]

    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])

    return np.stack(r_boxes)


# 把矩形变成正方形
def convert_to_square(bbox):
    square_bbox = bbox.copy()  # 复制一份框出来
    if bbox.shape[0] == 0:  # 批次取完 返回空
        return np.array([])

    # 求出矩形的宽和高
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]

    max_side = np.maximum(w, h)  # 求出最长边
    # 用宽和高的中心点 去扩充 两边均匀  square_bbox[:,0] = 最长边的中点减去宽的中点 加上bbox[:,0]
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side  # [x1,y1,x2,y2]
    square_bbox[:, 3] = square_bbox[:, 1] + max_side  # y2 = 扩充后的y1加上最长边

    return square_bbox


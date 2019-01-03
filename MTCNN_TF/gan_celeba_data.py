import numpy as np
from PIL import Image
import os
import traceback  # 回溯
from tool import utils

anno_src = r""
img_dir = ""

save_path = ""

for face_size in [12, 24, 48]:
    print("gan %i image" % face_size)

    # 图片保存路径
    positive_image_dir = os.path.join(save_path, str(face_size), "positive")
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")
    part_image_dir = os.path.join(save_path, str(face_size), "part")

    for dir_path in [save_path, positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 图片标签保存路径
    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0

    try:
        positive_anno_file = open(positive_anno_filename, "w")
        negative_anno_file = open(negative_anno_filename, "w")
        part_anno_file = open(part_anno_filename, "w")

        for i, line in enumerate(anno_src):
            if i < 2:
                continue
            try:
                strs = line.strip().split()
                image_filename = strs[0].strip()
                print("image_filename", image_filename)
                image_file = os.path.join(img_dir, image_filename)

                with Image.open(image_file) as img:
                    img_w, img_h = img.size
                    x1 = float(strs[1].strip())
                    y1 = float(strs[2].strip())
                    w = float(strs[3].strip())
                    h = float(strs[4].strip())
                    x2 = float(x1 + w)
                    y2 = float(y1 + h)

                    if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue

                    boxes = [[x1, y1, x2, y2]]

                    # 计算人脸中心点
                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    for _ in range(5):
                        w_ = np.random.randint(int(-w * 0.2), int(w * 0.2))
                        h_ = np.random.randint(int(-h * 0.2), int(h * 0.2))
                        # 把框变成正方形
                        side_len = np.random.randint(int(min(w, h) * 0.8), np.ceil(max(w, h) * 1.25))
                        x1_ = np.max(x1 + w / 2 + w_ - side_len / 2, 0)
                        y1_ = np.max(y1 + h / 2 + h_ - side_len / 2, 0)
                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len

                        crop_box = np.array([x1_, y1_, x2_, y2_])

                        # 计算坐标的偏移量
                        offset_x1 = (x1 - x1_) / side_len
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len

                        # 剪切图片，变成需要的face_size
                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size, face_size))

                        iou = utils.iou(crop_box, np.array(boxes))[0]
                        if iou > 0.65:
                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                    positive_count, 1, offset_x1, offset_y1, offset_x2, offset_y2
                                )
                            )
                            positive_anno_file.flush()
                            face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                            positive_count += 1
                        elif iou > 0.4:
                            part_anno_file.write(
                                "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                    negative_count, 2, offset_x1, offset_y1, offset_x2, offset_y2
                                )
                            )
                            part_anno_file.flush()
                            face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            part_count += 1
                        elif iou < 0.3:
                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0)
                            )
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1
                        _boxes = np.array(boxes)
                    for i in range(5):
                        side_len = np.random.randint(face_size, min(img_w, img_h) / 2)
                        x_ = np.random.randint(0, img_w - side_len)
                        y_ = np.random.randint(0, img_h - side_len)
                        crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])

                        if np.max(utils.iou(crop_box, _boxes)) < 0.3:
                            face_crop = img.crop(crop_box)
                            face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                            negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1

            except Exception as e:
                traceback.print_exc()
    finally:
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()

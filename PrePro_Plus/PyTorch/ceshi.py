import colorsys
import os
import time
from tqdm import tqdm
import numpy as np
# import onnxruntime
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET

from nets.ssd import SSD300
from utils.utils_map import get_coco_map, get_map
from utils.anchors import get_anchors
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from utils.utils_bbox import BBoxUtility


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的NUM_CLASSES、
#   model_path和classes_path参数的修改
# --------------------------------------------#
class SSD(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path": 'model_data/ssd_ship.pth',
        "classes_path": 'model_data/ship.txt',
        # ---------------------------------------------------------------------#
        #   用于预测的图像大小，和train时使用同一个即可
        # ---------------------------------------------------------------------#
        "input_shape": [300, 300],
        # -------------------------------#
        #   主干网络的选择
        #   vgg或者mobilenetv2或者resnet50
        # -------------------------------#
        "backbone": "resnet50",
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.01,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.45,
        # ---------------------------------------------------------------------#
        #   用于指定先验框的大小
        # ---------------------------------------------------------------------#
        'anchors_size': [30, 60, 111, 162, 213, 264, 315],
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": False,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化faster RCNN
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # ---------------------------------------------------#
        #   计算总的类的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors = torch.from_numpy(get_anchors(self.input_shape, self.anchors_size, self.backbone)).type(
            torch.FloatTensor)
        if self.cuda:
            self.anchors = self.anchors.cuda()
        self.num_classes = self.num_classes + 1

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.bbox_util = BBoxUtility(self.num_classes)
        self.generate()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   载入模型
    # ---------------------------------------------------#
    def generate(self, ifonnx=False):
        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        self.net = SSD300(self.num_classes, self.backbone)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        if not ifonnx:
            if self.cuda:
                # self.net = torch.nn.DataParallel(self.net)
                self.net = self.net.cuda()

        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        # self.net = onnx.load(self.model_path)
        self.net = self.net.eval()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, save=True, crop=False, count=False):
        top_list, left_list, bottom_list, right_list, class_list = [], [], [], [], []
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            # ---------------------------------------------------#
            #   转化成torch的形式
            # ---------------------------------------------------#
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            # -----------------------------------------------------------#
            #   将预测结果进行解码
            # -----------------------------------------------------------#
            results = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape,
                                                self.letterbox_image,
                                                nms_iou=self.nms_iou, confidence=self.confidence)
            # --------------------------------------#
            #   如果没有检测到物体，则返回原图
            # --------------------------------------#
            if len(results[0]) <= 0:
                return image, [0], 0

            if save:
                data = np.array(results[0], dtype='float32')

                # Grouping by class (column 5)
                grouped_data = {}
                for row in data:
                    class_label = int(row[4].item())
                    if class_label not in grouped_data:
                        grouped_data[class_label] = []
                    grouped_data[class_label].append(row)

                # Calculate the area for each box and find the median
                medians = {}
                for class_label, boxes in grouped_data.items():
                    boxes = np.array(boxes)
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    median_idx = np.argsort(areas)[len(areas) // 2]
                    medians[class_label] = boxes[median_idx]
                results = np.stack(list(medians.values()))

                top_label = np.array(results[:, 4], dtype='int32')  # 相当于是类型，预测的属于哪一类
                top_conf = results[:, 5]  # 这个是置信度
                top_boxes = results[:, :4]  # 这个相当于是边框位置
                top_list = results[:, 6:]
            else:
                top_label = np.array(results[0][:, 4], dtype='int32')  # 相当于是类型，预测的属于哪一类
                top_conf = results[0][:, 5]  # 这个是置信度
                top_boxes = results[0][:, :4]  # 这个相当于是边框位置
                top_list = results[0][:, 6:]
        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)
        # ---------------------------------------------------------#
        #   计数
        # ---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        # ---------------------------------------------------------#
        #   是否进行目标的裁剪
        # ---------------------------------------------------------#
        # if crop:
        #     for i, c in list(enumerate(top_boxes)):
        #         top, left, bottom, right = top_boxes[i]
        #         top = max(0, np.floor(top).astype('int32'))
        #         left = max(0, np.floor(left).astype('int32'))
        #         bottom = min(image.size[1], np.floor(bottom).astype('int32'))
        #         right = min(image.size[0], np.floor(right).astype('int32'))
        #
        #         dir_save_path = "img_crop"
        #         if not os.path.exists(dir_save_path):
        #             os.makedirs(dir_save_path)
        #         crop_image = image.crop([left, top, right, bottom])
        #         crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
        #         print("save crop_" + str(i) + ".png to " + dir_save_path)
        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
            # top_list.append(top)
            # left_list.append(left)
            # bottom_list.append(bottom)
            # right_list.append(right)
            # class_list.append(c)

        return image, top_boxes, top_conf

    def get_FPS(self, image, test_interval):
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            # ---------------------------------------------------#
            #   转化成torch的形式
            # ---------------------------------------------------#
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            # -----------------------------------------------------------#
            #   将预测结果进行解码
            # -----------------------------------------------------------#
            results = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape,
                                                self.letterbox_image,
                                                nms_iou=self.nms_iou, confidence=self.confidence)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                # ---------------------------------------------------------#
                outputs = self.net(images)
                # -----------------------------------------------------------#
                #   将预测结果进行解码
                # -----------------------------------------------------------#
                results = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape,
                                                    self.letterbox_image,
                                                    nms_iou=self.nms_iou, confidence=self.confidence)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            # ---------------------------------------------------#
            #   转化成torch的形式
            # ---------------------------------------------------#
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            # -----------------------------------------------------------#
            #   将预测结果进行解码
            # -----------------------------------------------------------#
            results = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape,
                                                self.letterbox_image,
                                                nms_iou=self.nms_iou, confidence=self.confidence)
            # --------------------------------------#
            #   如果没有检测到物体，则返回原图
            # --------------------------------------#
            if len(results[0]) <= 0:
                return

            top_label = np.array(results[0][:, 4], dtype='int32')
            top_conf = results[0][:, 5]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return


if __name__ == "__main__":
    '''
    Recall和Precision不像AP是一个面积的概念，因此在门限值（Confidence）不同时，网络的Recall和Precision值是不同的。
    默认情况下，本代码计算的Recall和Precision代表的是当门限值（Confidence）为0.5时，所对应的Recall和Precision值。

    受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算不同门限条件下的Recall和Precision值
    因此，本代码获得的map_out/detection-results/里面的txt的框的数量一般会比直接predict多一些，目的是列出所有可能的预测框，
    '''
    # --------------------------------------------------------------------------------------#
    #   此处的classes_path用于指定需要测量VOC_map的类别
    #   一般情况下与训练和预测所用的classes_path一致即可
    # --------------------------------------------------------------------------------------#
    classes_path = 'model_data/ship.txt'
    # --------------------------------------------------------------------------------------#
    #   MINOVERLAP用于指定想要获得的mAP0.x，mAP0.x的意义是什么请同学们百度一下。
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    #
    #   当某一预测框与真实框重合度大于MINOVERLAP时，该预测框被认为是正样本，否则为负样本。
    #   因此MINOVERLAP的值越大，预测框要预测的越准确才能被认为是正样本，此时算出来的mAP值越低，
    # --------------------------------------------------------------------------------------#
    MINOVERLAP = 0.5
    # --------------------------------------------------------------------------------------#
    #   受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算mAP
    #   因此，confidence的值应当设置的尽量小进而获得全部可能的预测框。
    #
    #   该值一般不调整。因为计算mAP需要获得近乎所有的预测框，此处的confidence不能随便更改。
    #   想要获得不同门限值下的Recall和Precision值，请修改下方的score_threhold。
    # --------------------------------------------------------------------------------------#
    confidence = 0.001
    # --------------------------------------------------------------------------------------#
    #   预测时使用到的非极大抑制值的大小，越大表示非极大抑制越不严格。
    #
    #   该值一般不调整。
    # --------------------------------------------------------------------------------------#
    nms_iou = 0.5
    # ---------------------------------------------------------------------------------------------------------------#
    #   Recall和Precision不像AP是一个面积的概念，因此在门限值不同时，网络的Recall和Precision值是不同的。
    #
    #   默认情况下，本代码计算的Recall和Precision代表的是当门限值为0.5（此处定义为score_threhold）时所对应的Recall和Precision值。
    #   因为计算mAP需要获得近乎所有的预测框，上面定义的confidence不能随便更改。
    #   这里专门定义一个score_threhold用于代表门限值，进而在计算mAP时找到门限值对应的Recall和Precision值。
    # ---------------------------------------------------------------------------------------------------------------#
    score_threhold = 0.5
    # -------------------------------------------------------#
    #   map_vis用于指定是否开启VOC_map计算的可视化
    # -------------------------------------------------------#
    map_vis = False
    # -------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    # -------------------------------------------------------#
    VOCdevkit_path = '/data/Newdisk/zhaoxiaoming/steal/sjs/yolox_new/SHIP'
    # -------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    # -------------------------------------------------------#
    map_out_path = 'map_out'

    image_ids = open(os.path.join(VOCdevkit_path, "Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    start_time = time.time()
    class_names, _ = get_classes(classes_path)
    print("Load model.")
    # key = input('Encode key:')
    value = input('Input value:')
    ssd = SSD(confidence=confidence, nms_iou=nms_iou)
    # if key2==yolo.key:
    print("Load model done.")
    print("Get predict result.")
    for image_id in tqdm(image_ids):
        image_path = os.path.join(VOCdevkit_path, "jpg/" + image_id + ".jpg")
        image = Image.open(image_path)
        if map_vis:
            image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
        ssd.get_map_txt(image_id, image, class_names, map_out_path)
    print("Get predict result done.")
    print("Get ground truth result.")
    for image_id in tqdm(image_ids):
        with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
            root = ET.parse(os.path.join(VOCdevkit_path, "xml/" + image_id + ".xml")).getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult') != None:
                    difficult = obj.find('difficult').text
                    if int(difficult) == 1:
                        difficult_flag = True
                obj_name = obj.find('name').text
                if obj_name not in class_names:
                    continue
                bndbox = obj.find('bndbox')
                left = bndbox.find('xmin').text
                top = bndbox.find('ymin').text
                right = bndbox.find('xmax').text
                bottom = bndbox.find('ymax').text

                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    print("Get ground truth result done.")
    print("Get map.")
    get_map(MINOVERLAP, True, score_threhold=score_threhold, path=map_out_path)
    print("Get map done.")
    end_time = time.time()  # 结束运行时间
    print('cost %f second' % (end_time - start_time))

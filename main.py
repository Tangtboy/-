#!/user/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import random
import time
from pathlib import Path

import keyboard
import torch
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QApplication, QFileDialog, QMessageBox
from torch.backends import cudnn

import detect
from UI import MainUI
import os
import sys
import cv2
from PyQt5.QtGui import QIcon, QPixmap

from utils.general import check_requirements, strip_optimizer
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.main_ui = MainUI.Ui_Object()
        self.main_ui.setupUi(self)
        self.path = None
        self.save_dir = None
        self.img = None
        # self.timer = QTimer(self)
        self.interrupt = False
        self.detect_time = 0
        self.detect_number = 0
        # self.main_dataset = None
        self.slot()

    def slot(self):
        self.main_ui.pushButton.clicked.connect(self.Model_Choice)
        self.main_ui.pushButton_2.clicked.connect(self.Image_Choice)
        self.main_ui.pushButton_3.clicked.connect(self.Image_Detection)
        self.main_ui.pushButton_4.clicked.connect(self.Result_Dispaly)
        self.main_ui.pushButton_5.clicked.connect(self.Result_Save)
        self.main_ui.pushButton_6.clicked.connect(self.ImageDetct_End)
        self.main_ui.pushButton_7.clicked.connect(self.Media_Choice)
        self.main_ui.pushButton_8.clicked.connect(self.Media_Detection)
        self.main_ui.pushButton_11.clicked.connect(self.Media_Display)
        self.main_ui.pushButton_13.clicked.connect(self.Camera_Detection)
        # self.timer.timeout.connect(self.Camera_Display)
        self.main_ui.pushButton_14.clicked.connect(self.Camera_Close)

    def Model_Choice(self):
        txt = self.main_ui.comboBox_2.currentText()
        print("选择了" + txt)
        if txt == "best.pt":
            param = "best.pt"
            print("选择了best.pt")
            return param
        elif txt == "yolov5s.pt":
            param = "yolov5s.pt"
            print("选择了yolov5s.pt")
            return param

    def Image_Choice(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'All Files (*)', options=options)
        if file_path:
            self.main_ui.label_10.clear()
            print(f'Selected File: {file_path}')
            pixmap = QPixmap(file_path)  # 读取图片
            # pixmap = pixmap.scaled(self.main_ui.label.width(), self.main_ui.label.height())  # 缩放图片
            self.main_ui.label_10.setPixmap(pixmap)  # 显示图片
            self.main_ui.label_10.setScaledContents(True)  # 让图片自适应label大小
            self.path = file_path

    def Image_Detection(self):
        param = self.Model_Choice()
        file_path = self.path
        self.save_dir, self.detect_time, self.detect_number = detect.parser_set(param, file_path)

    def Result_Dispaly(self):
        # cv2.imshow("result", self.im0)
        # cv2.waitKey(0)
        img = cv2.imread(self.save_dir)
        self.img = img
        height, width, channel = img.shape
        bytes_per_line = 3 * width  # 每行的字节数
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)  # 读取图片
        self.main_ui.label_10.setPixmap(pixmap)  # 显示图片
        self.main_ui.label_10.setScaledContents(True)  # 让图片自适应label大小
        self.main_ui.lineEdit.setText(str(self.detect_time) + "s")
        self.main_ui.lineEdit_2.setText(str(self.detect_number))

    def Result_Save(self):
        path = 'F:/Pythonproject_main/Car_detect/result'
        cv2.imwrite(os.path.join(path, 'result.jpg'), self.img)

    def ImageDetct_End(self):
        self.main_ui.label_10.clear()

    def Media_Choice(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'All Files (*)', options=options)
        self.path = file_path
        if file_path:
            video = cv2.VideoCapture(file_path)
            # 获取输入视频的宽度
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            # 获取输入视频的高度
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # 获取视频帧数
            frame_number = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            # 获取输入视频的帧率
            frame_rate = int(video.get(cv2.CAP_PROP_FPS))

            ratio1 = width / 500  # (label 宽度)
            ratio2 = height / 500  # (label 高度)
            ratio = max(ratio1, ratio2)

            while video.isOpened():
                ret, frame = video.read()
                # 将图片转换为 Qt 格式
                # QImage:QImage(bytes,width,height,format)
                picture = QImage(frame, width, height, 3 * width, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(picture)
                # 按照缩放比例自适应 label 显示
                pixmap.setDevicePixelRatio(ratio)
                self.main_ui.label_10.setPixmap(pixmap)
                self.main_ui.label_10.setScaledContents(True)
                cv2.waitKey(10)
                if not ret:
                    break

            print("结束了")

    def Media_Detection(self):
        param = self.Model_Choice()
        file_path = self.path
        self.save_dir, self.detect_time, self.detect_number = detect.parser_set(param, file_path)
        self.main_ui.lineEdit.setText(str(self.detect_time) + "s")
        self.main_ui.lineEdit_2.setText(str(self.detect_number))

    def Media_Display(self):
        if self.save_dir:
            video = cv2.VideoCapture(self.save_dir)
            # 获取输入视频的宽度
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            # 获取输入视频的高度
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # 获取视频帧数
            frame_number = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            # 获取输入视频的帧率
            frame_rate = int(video.get(cv2.CAP_PROP_FPS))

            ratio1 = width / 500  # (label 宽度)
            ratio2 = height / 500  # (label 高度)
            ratio = max(ratio1, ratio2)

            while video.isOpened():
                ret, frame = video.read()
                # 将图片转换为 Qt 格式
                # QImage:QImage(bytes,width,height,format)
                picture = QImage(frame, width, height, 3 * width, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(picture)
                # 按照缩放比例自适应 label 显示
                pixmap.setDevicePixelRatio(ratio)
                self.main_ui.label_10.setPixmap(pixmap)
                self.main_ui.label_10.setScaledContents(True)
                cv2.waitKey(10)
                if not ret:
                    break
            print("结束了")

    def Camera_Display(self, img):
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色空间
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.main_ui.label_10.setPixmap(pixmap)

    def detect1(self, opt):
        self.interrupt = False
        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
            # self.main_dataset = dataset
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
            # self.main_dataset = dataset
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                if view_img:
                    self.Camera_Display(im0)
                    # cv2.imshow(str(p), im0)
                    cv2.waitKey(1)
                    dataset.interrupted = self.interrupt
                    # cv2.waitKey(1)  # 1 millisecond
                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path += '.mp4'
                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            vid_writer.write(im0)
                        #
                        print(str(p))

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')

    def Camera_Detection(self):
        weight = self.Model_Choice()
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=weight, help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        opt = parser.parse_args()
        check_requirements(exclude=('pycocotools', 'thop'))

        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                    self.detect1(opt)
                    strip_optimizer(opt.weights)
            else:
                self.detect1(opt)

    def Camera_Close(self):
        # self.timer.stop()
        self.interrupt = True
        self.main_ui.label_10.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())

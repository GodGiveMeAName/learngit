
import argparse
import time
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *


def detect(
        cfg,
        data_cfg,
        weights,
        images='data/samples',  # input folder
        output='output',  # output folder
        fourcc='mp4v',
        img_size=412,
        conf_thres=0.5,
        nms_thres=0.3,
        save_txt=False,
        save_images=True,
        webcam=False
):
    device = torch_utils.select_device()
    torch.backends.cudnn.benchmark = False  # set False for reproducible results
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    if ONNX_EXPORT:
        s = (320, 192)  # (320, 192) or (416, 256) or (608, 352) onnx model image size (height, width)
        model = Darknet(cfg, s)
    else:
        model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    model.fuse()

    # Eval mode
    model.to(device).eval()

    if ONNX_EXPORT:
        img = torch.zeros((1, 3, s[0], s[1]))
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    for i, (path, img, im0, vid_cap) in enumerate(dataloader):
        t = time.time()
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred, _ = model(img)
        det = non_max_suppression(pred, conf_thres, nms_thres)[0]

        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results to screen
            print('%gx%g ' % img.shape[2:], end='')  # print image size
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            cls_flag = False    # 用来判断是否保存该类别的样本（针对水瓶做的，水瓶样本太多了）
            for *xyxy, conf, cls_conf, cls in det:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
                if not cls_flag:
                    if classes[int(cls)] != "liquid":           # 如果含有违禁品 且不是水瓶 那么 标签位置为 True
                        print(classes[int(cls)]+'_')
                        cls_flag = True

                # Add bbox to the image
                if conf > 0.5:
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        print('Done. (%.3fs)' % (time.time() - t))

        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)

        if save_images:  # Save image with detections
            if dataloader.mode == 'images':
                #if det is not None: #and cls_flag:     # 如果含有违禁品 且不是水瓶 标签位置为 True 则保存
                cv2.imwrite(save_path, im0)
                #else:
                 #   cv2.imwrite(r'E:\\data\\Tianhe_20190911_n', im0)#
            else:
                if vid_path != save_path:     # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
                vid_writer.write(im0)

    if save_images:
        print('Results saved to %s' % os.getcwd() + os.sep + output)
        if platform == 'darwin':  # macos
            os.system('open ' + output + ' ' + save_path)

class Detector(object):
    """docstring for Detector"""
    def __init__(self, cfg, data_cfg, weights, output='output', img_size=416, conf_thres=0.5, nms_thres=0.5):
        super(Detector, self).__init__()
        # Initialize model
        self.model = Darknet(cfg, img_size)
        self.classes = load_classes(parse_data_cfg(data_cfg)['names'])
        self.output = output
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.device = torch_utils.select_device()
        self.colors = [[0,0,255], [0,255,0], [255,0,0], [0,255,255], [255,0,255],[125,125,125],[125,125],[0,125,255],[255,125,125]]
        if not os.path.exists(output):
            os.makedirs(output)
        
        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(weights, map_location=self.device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(self.model, weights)

        self.model.to(self.device).eval()
        print("okokok")

    def detect_one_image(self, image_path):
        im0 = cv2.imread(image_path)  # BGR
        assert im0 is not None, 'File Not Found ' + path
        # Padded resize
        img, _, _, _ = letterbox(im0, new_shape=self.img_size)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        save_path = os.path.join(self.output, os.path.basename(image_path))
        print(save_path)
        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        pred, _ = self.model(img)
        detections = non_max_suppression(pred, self.conf_thres, self.nms_thres)[0]
        if detections is not None and len(detections) > 0:
            # Rescale boxes from 416 to true image size
            scale_coords([self.img_size,self.img_size], detections[:, :4], im0.shape).round()

            # Print results to screen
            for c in detections[:, -1].unique():
                n = (detections[:, -1] == c).sum()
                # print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in detections:
                # Add bbox to the image
                # label = '%s %.2f' % (classes[int(cls)], conf)
                label = '%s'%self.classes[int(cls)]
                plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)],line_thickness=2)

            cv2.imwrite(save_path, im0)
        return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny-8cls.cfg', help='cfg file path')
    #parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--data-cfg', type=str, default='data/rbc.data', help='coco.data file path')
    #parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default=r'data\\samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=736, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.1, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='fourcc output video codec (verify ffmpeg support)')
    parser.add_argument('--output', type=str, default=r'outputs', help='specifies the output path for images and videos')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt.cfg,
               opt.data_cfg,
               opt.weights,
               images=opt.images,
               img_size=opt.img_size,
               conf_thres=opt.conf_thres,
               nms_thres=opt.nms_thres,
               fourcc=opt.fourcc,
               output=opt.output)

import argparse
import time
import datetime
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *
import os.path as osp
from ipdb import set_trace


def detect(
        cfg,
        data_cfg,
        weights,
        images,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
        save_txt=False,
        save_images=False,
        webcam=False
):
    device = torch_utils.select_device()
    # if os.path.exists(output):
        # shutil.rmtree(output)  # delete output folder
    # os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    model.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    colors = [[0,0,255], [0,255,0], [255,0,0], [0,255,255], [255,0,255],[125,125,125],[125,125],[0,125,255],[255,125,125]]
    start = time.time()
    # print(colors)
    cv2.namedWindow("Detection Result:[Press 'ESC' to Quit]",flags = cv2.WINDOW_NORMAL)   # ----- Added by ZWX
    cv2.namedWindow("Raw Image:[Press 'ESC' to Quit]",flags = cv2.WINDOW_NORMAL)          # ----- Added by ZWX
    for j, (path, img, im0, vid_cap) in enumerate(dataloader):  # img is the pic after trans, im0 is raw pic.
        t = time.time()
        save_path = str(Path(output) / Path(path).name)
        cv2.imshow("Raw Image:[Press 'ESC' to Quit]", im0)      # -----Added by ZWX
        #print(im0.shape)
        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return
        pred, _ = model(img)
        detections = non_max_suppression(pred, conf_thres, nms_thres)[0]
        
        flag_save_img = 0 # 记录是否保存当前帧
        if detections is not None and len(detections) > 0:
            # Rescale boxes from 416 to true image size
            scale_coords([img_size,img_size], detections[:, :4], im0.shape).round()

            # Print results to screen
            for c in detections[:, -1].unique():
                n = (detections[:, -1] == c).sum()
                # print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in detections:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                # Add bbox to the image
                # label = '%s %.2f' % (classes[int(cls)], conf)
                label = '%s'%classes[int(cls)]
                flag_save_img = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)],line_thickness=2) or flag_save_img

        print('Done. (%.3fs)' % (time.time() - t))   #'''打印当前帧计算时间'''
        # cv2.imshow(path, im0)
        cv2.imshow("Detection Result:[Press 'ESC' to Quit]", im0)       #cv2.imshow(weights, im0) ----- Modedied by ZWX
        cv2.waitKey(1)  #  ----- Modedied by ZWX

        if webcam:  # Show live webcam
            if flag_save_img:           # 保存违禁品图片到本地
                print('----------------------------********************----------------------')
                data_now = datetime.datetime.now().strftime('%Y-%m-%d')
                time_now = datetime.datetime.now().strftime('%H-%M-%S')
                cur_img_save_path = "detection result\\"+str(data_now)
                if not os.path.exists(cur_img_save_path):
                    os.makedirs(cur_img_save_path)
                cv2.imwrite(cur_img_save_path + '\\' + str(time_now) + '.png', im0)
                print("save")
            

        if save_images:  # Save generated image with detections
            # print('000000')
            if dataloader.mode == 'video':
                # print('111111')
                if vid_path != save_path:  # new video
                    # print('222222')
                    print(save_path)
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    codec = int(vid_cap.get(cv2.CAP_PROP_FOURCC))
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    #vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), 3, (width, height))
                vid_writer.write(im0)

            else:
                cv2.imwrite(save_path, im0)
    # cv2.waitKey(0)
    # cv2.destroyAllwindows()
    print('cost time is %.3fs' % (time.time()-start))

    # if save_images and platform == 'darwin':  # macos
    #     os.system('open ' + output + ' ' + save_path)


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
        self.colors = [[0,0,255], [0,255,0], [255,0,0], [0,255,255], [255,0,255]]
        if not osp.exists(output):
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

        save_path = osp.join(self.output, osp.basename(image_path))
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

    

def detect_one_image(
        cfg,
        data_cfg,
        weights,
        image_path,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
):
    device = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    model.to(device).eval()
    
    # dataloader = LoadImages(images, img_size=img_size)
    im0 = cv2.imread(image_path)  # BGR
    assert im0 is not None, 'File Not Found ' + path
    # Padded resize
    img, _, _, _ = letterbox(im0, new_shape=img_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    colors = [[0,0,255], [0,255,0], [255,0,0], [0,255,255], [255,0,255],[125,125,125],[125,125],[0,125,255],[255,125,125]]

    save_path = osp.join(output, osp.basename(image_path))
    # Get detections
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    pred, _ = model(img)
    detections = non_max_suppression(pred, conf_thres, nms_thres)[0]
    if detections is not None and len(detections) > 0:
        # Rescale boxes from 416 to true image size
        scale_coords([img_size,img_size], detections[:, :4], im0.shape).round()

        # Print results to screen
        for c in detections[:, -1].unique():
            n = (detections[:, -1] == c).sum()
            # print('%g %ss' % (n, classes[int(c)]), end=', ')

        # Draw bounding boxes and labels of detections
        for *xyxy, conf, cls_conf, cls in detections:
            # Add bbox to the image
            # label = '%s %.2f' % (classes[int(cls)], conf)
            label = '%s'%classes[int(cls)]
            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)],line_thickness=2)

        cv2.imwrite(save_path, im0)
    return save_path

    # if save_images and platform == 'darwin':  # macos
    #     os.system('open ' + output + ' ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg\yolov3-tiny-8cls.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/rbc.data', help='.data file path')
    parser.add_argument('--weights', type=str, default='weights\\latest.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default=r'data\2019-09-11 12-24-00.mp4', help='path to images')
    parser.add_argument('--img-size', type=int, default=608, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.1, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.data_cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            webcam=False
        )

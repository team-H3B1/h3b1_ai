import asyncio
import base64
import datetime
import time
from threading import Thread

import torch
from torch.backends import cudnn
from torchvision.transforms import GaussianBlur

from deep_sort.deep_sort import DeepSort
import cv2
from pathlib import Path
import os
import requests
import sys
from knn.Mediapipe import MediaPipe

sys.path.insert(0, './yolov5')
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from PIL import Image
import _thread
import requests


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

counting_obj1 = set() 
mediapipe = MediaPipe()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# DEVICE1 = select_device('')
# MODEL1 = DetectMultiBackend('ai.pt', device=DEVICE1, dnn=False)
# STRIDE1, NAMES1, PT1, JIT1, _ = MODEL1.stride, MODEL1.names, MODEL1.pt, MODEL1.jit, MODEL1.onnx
# IMG_SIZE1 = check_img_size([640, 640], s=STRIDE1)  # check image size
# ip_camera1 = LoadStreams('https://www.youtube.com/watch?v=BBdC1rl5sKY', img_size=IMG_SIZE1, stride=STRIDE1, auto=PT1 and not JIT1)


def stream(source):
    print(torch.cuda.is_available())
    yolo_model = 'yolov5s.pt'
    deep_sort_model = 'osnet_x0_25'

    image_size = [640, 640]
    conf_thres = 0.3
    iou_thres = 0.5
    device = '0'
    classes = None
    agnostic_nms = False
    deepsort_config = 'deep_sort/configs/deep_sort.yaml'
    half = False
    max_det = 1000
    dnn = True

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(deepsort_config)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    image_size = check_img_size(image_size, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=image_size, stride=stride, auto=pt and not jit)
    else:
        dataset = LoadImages(source, img_size=image_size, stride=stride, auto=pt and not jit)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    if pt and device.type != 'cpu':
        print("this program use CUDA")
        model(torch.zeros(1, 3, *image_size).to(device).type_as(next(model.model.parameters())))  # warmup

    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(img, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            w, h = im0.shape[1], im0.shape[0]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} , "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0) # 이거 cpu cuda()로 바꿔봐야 할 듯 => 에러남.. what?
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]  # 박스 꼭짓점 들 인듯?
                        id = output[4]  # 아마도 객체 아이디
                        cls = output[5]  # class 이다. 욜로 프리트레인모델 기준 0은 person
                        c = int(cls)  # integer class
                        #######################
                            # count
                        count_obj(bboxes, w, h, id, c, im0)
                        ########################

                        if c == 0 or c == 80:
                            label = f'{"사람" if cls == 0 else "화재"} {id} {conf:.2f}'
                            annotator.box_label(bboxes, label, color=colors(c, True))

                        # #############
                        # blur
                        # print('bbox', bboxes)
                        # blurring = GaussianBlur((7, 13), sigma=(0.1, 0.2))
                        # # blur_img = blurring(torch.tensor(bboxes))
                        # blur_img = blurring(torch.tensor(bboxes))
                        # cv2.imshow(str(p), blur_img)
                        # #############

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort.increment_ages()
                # LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            #
            # if show_vid:
            # global count
            color = (0, 255, 0)
            # draw vertical line
            # cv2.putText(im0, str(t.get_count()), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            start_point = (w - 700, h - 600)
            end_point = (w - 700, h)
            cv2.line(im0, start_point, end_point, color, thickness=2)


        image_bytes = cv2.imencode('.jpg', im0)[1].tobytes()
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n'


# class 하나 만들어서 변수 관리

# def get_detect_img(img, box):
#     b64_str1 = base64.b64encode(image1[1]).decode('utf-8')

#     color = (0, 255, 0)
#     image2 = cv2.imencode('.jpg', cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness=1))
#     b64_str2 = base64.b64encode(image2[1]).decode('utf-8')
#     # print(b64_str2)

#     data = {
#         "b64_str1": b64_str1,
#         "b64_str2": b64_str2
#     }
#     return data

def wait10(id):
    time.sleep(10)
    counting_obj1.add(id)

url = ""
def request_cv(msg):
    data = {
        'message' : msg
    }
    try :
        http_post_req = requests.post(url, json=data)
        if http_post_req.status_code == 200 :
            LOGGER.info(f'아 암튼 됨 ㅅㄱ')
        else :
            LOGGER.error(F'sadf')
    except Exception:
        LOGGER.error('서버 통신 중 에러 발생')

def count_obj(box, w, h, id, cls, img):
    if cls == 0 :
        image1 = cv2.imencode('.jpg', img[box[1]:box[3], box[0]:box[2]])
        image2, state = mediapipe.getPose(image1)
        LOGGER.info(f"{id}'s state is {state}")
        if state == "fall" :
            _thread.start_new_thread(wait10,id,)
    if cls == 80 and not id in counting_obj1 :
        counting_obj1.add(id)
        # 화재 발생
        request_cv("화재가 감지되었습니다.")
    
    if cls == 0 and id in counting_obj1 :
        request_cv("쓰러짐이 감지되었습니다.")
        counting_obj1.remove(id)


import argparse
from ultralytics import YOLO


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/yolov8n-face.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', nargs='+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--dynamic', action='store_true', default=False, help='enable dynamic axis in onnx model')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    model = YOLO(opt.weights)
    model.export(
        format='onnx',
        imgsz=opt.img_size,
        dynamic=opt.dynamic,
        device=opt.device,
        opset=17
    )
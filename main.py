import os
os.environ['YOLO_VERBOSE'] = str(False)
from ultralytics import YOLO
import cv2

# 17个关键点连接顺序
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6]] #[5, 7]，这个点位可以放在最后，也可以不加，本代码暂时不加

def get_class(bbox,cls, conf):
    dic_result = {}
    dic_result["bbox"] = bbox
    dic_result["cls"] = cls
    dic_result["conf"] = conf
    return dic_result


def infer_pose(image, re_list):
    # 创建一个存储x和y坐标的列表
    x_values = [0] * 17
    y_values = [0] * 17

    for re in re_list:
        if hasattr(re, 'keypoints'):
            keypoints = re.keypoints
            data = keypoints.data
            for key in data:
                for i, point in enumerate(key.tolist()):
                    x_values[i] = point[0]
                    y_values[i] = point[1]

                # 绘制坐标点
                for i in range(len(x_values)):
                    cv2.circle(image, (int(x_values[i]), int(y_values[i])), 5, (0, 0, 255), -1)  # 绘制红色的圆点

                # 绘制连接线
                for connection in skeleton:
                    start_point = (int(x_values[connection[0] - 1]), int(y_values[connection[0] - 1]))
                    end_point = (int(x_values[connection[1] - 1]), int(y_values[connection[1] - 1]))
                    cv2.line(image, start_point, end_point, (0, 255, 0), thickness=2)  # 绘制绿色的连线
        else:
            return image
    return image
def infer(re):
    dic = []
    if not re:  # 如果re为空，返回空列表
        return dic

    for result in re:
        boxes = result.boxes  # 用于边界框输出的Boxes对象
        for box in boxes:
            x_min, y_min, x_max, y_max, conf, cls = box.data[0].tolist()
            bbox = x_min, y_min, x_max, y_max
            result_dic = get_class(bbox,cls,conf)
            dic.append(result_dic)
    return dic

def draw_img(names,img,dic_list):
    if not dic_list:
        return img
    for result in dic_list:
        x_min, y_min, x_max, y_max = result["bbox"]
        cls = int(result["cls"])
        conf = result["conf"]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,255,0), 2)
        text =f"{names[cls]} "f'{conf:.2f}'
        cv2.putText(img, text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0,255,0), 2)
    return img

#图片识别
if __name__ == '__main__':
    weights = './weights/yolov8n-pose.pt'
    img_path = './images/bus.jpg'

    img = cv2.imread(img_path)
    model = YOLO(weights)
    names = model.names
    #device = cuda device, i.e. 0 or 0,1,2,3 or cpu
    results = model.predict(source=img,imgsz=640,conf=0.6,iou=0.5,device="cpu")

    image = img.copy()
    #img = infer_pose(image,results)   #pose姿态关键点，可用测试拓展

    dic_list = infer(results)
    img = draw_img(names,img,dic_list)

    cv2.imshow("1",img)
    cv2.waitKey(0)

#视频识别
# if __name__ == '__main__':
#     weights = './weights/yolov8n-pose.pt'
#     video_path = './video/1.mp4'
#
#     cap = cv2.VideoCapture(video_path)
#     model = YOLO(weights)
#     names = model.names
#     while cap.isOpened():
#         ret, frame = cap.read()
#
#         if not ret:
#             break
#         #device = cuda device, i.e. 0 or 0,1,2,3 or cpu
#         # 对每一帧进行推理
#         results = model.predict(source=frame, imgsz=640, conf=0.6, iou=0.5, device="cpu")
#
#         #frame = infer_pose(frame,results)  #pose姿态关键点，可用测试拓展
#
#         dic_list = infer(results)
#         frame = draw_img(names,frame, dic_list)
#
#         cv2.imshow('Video', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
# yolov8_infer / yolov8_face_infer

本项目适用于yolov8以及yolov8-face的相关官方模型推理，引出api接口方便后期调用和移植。

！！！注意！！！

代码引出yolov8n-pose.pt的相关api接口infer_pose(),如想进行相关调用，开启对应屏蔽代码即可，可进行相关测试及其修改，代码分问图片识别和视频识别，解开相关屏蔽代码即可。

```python
#图片识别测试
img = infer_pose(image,results)   #pose姿态关键点，可用测试拓展
#视频识别相关测试
frame = infer_pose(frame,results)  #pose姿态关键点，可用测试拓展

```

代码部署部分

```python
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

代码相关问题可以联系我

QQ：1957435942

vx：a15535592096
# modelServe.py
from http.server import BaseHTTPRequestHandler, HTTPServer
from functools import partial
import json
from Tools.visulize import ava_inference_transform
from utils.Data_process import build_final_data,imgs_bytes_list_to_clip,to_tensor
from utils.Format_conversion import Yolo5LikeDetections
from utils.logger import get_logger
import torch
import numpy as np
import redis
import pickle
import numpy as np
import time
import cv2

logs=get_logger("module", filename="modlelServe")
def process_one_clip(system, id_to_ava_labels, cfg,data_id):
    Data={}
    FPS = cfg.get('video', {}).get('fps')
    PERSON_CLASS_ID = cfg.get('PERSON_CLASS_ID')
    imsize = cfg.get('imsize')
    device = cfg.get('device')
    slowfast=cfg.get('slowfast')
    alpha=slowfast.get('alpha')
    yolo=cfg.get('yolo')
    yolo_iou=yolo.get('confidence_threshold')
    yolo_conf=yolo.get('iou_threshold')
# 1. 连接Redis并取出字节
    tt=time.time()
    redis_client = redis.Redis(host='127.0.0.1', port=6379, db=1)
    video_data = redis_client.get(f'data:images:{data_id}')
    imgs_data = pickle.loads(video_data)#type: ignore
    logs.info(f"extracting video data from redis cost:{1000*(time.time()-tt)}ms")
#yolo检测
    tt=time.time()
    res = system.yolo_model(imgs_data, imgsz=imsize, verbose=False,iou=yolo_iou,conf=yolo_conf)
    yolo_preds = Yolo5LikeDetections(res, imgs_data, system.yolo_model)
    logs.info(f"📷 processing YOLO耗时：{1000*(time.time()-tt)}")

# deepsort跟踪
    tx=time.time()
    deepsort_outputs = []
    for ifps in range(len(yolo_preds.pred)):
        temp = system.deepsort_update(
            yolo_preds.pred[ifps].cpu(),
            yolo_preds.xywh[ifps][:, 0:4].cpu(),
            yolo_preds.ims[ifps]
        )
        if len(temp) == 0:
            temp = np.ones((0, 8))
        deepsort_outputs.append(temp.astype(np.float32))
    yolo_preds.pred = deepsort_outputs
    logs.info(f"👁  deeosort time:{1000*(time.time()-tx)}")

    # 行为识别 slowfast
    tt=time.time()
    clip=imgs_bytes_list_to_clip(imgs_data, to_tensor)  
    detections = yolo_preds.pred[int(FPS/2)]  # [n,8] -> [x1, y1, x2, y2, cls, trackid, vx, vy]
    # 根据类别过滤出 person（人）的框
    person_mask = detections[:, 4] == PERSON_CLASS_ID
    person_boxes = detections[person_mask][:, :4] 
    # 如果至少有一个人
    if person_boxes.shape[0] > 0:
        inputs, inp_boxes, _ = ava_inference_transform(clip, person_boxes, 
                                                        crop_size=imsize,num_frames=FPS,
                                                        slow_fast_alpha=alpha,device=device)
        with torch.no_grad():
            slowfaster_preds = system.slowfast_model(inputs, inp_boxes.to(device))
            slowfaster_preds = slowfaster_preds.cpu()
            logs.info(f"📏 processing slowfast耗时：{1000*(time.time()-tt)}")
        for tid,avalabel in zip(detections[person_mask][:,5].tolist(), np.argmax(slowfaster_preds, axis=1).tolist()):
            id_to_ava_labels[tid] = system.ava_labelnames[avalabel+1]
    tt=time.time()
    # pic=system.save_yolopreds_tovideo(yolo_preds, id_to_ava_labels,save=False)
    pic =np.random.randint(0,255,(480,640,3))#填充虚假图片
    Data = build_final_data(yolo_preds.pred[int(FPS/2)], id_to_ava_labels,pic)
    logs.info(f"📡 processing export json耗时：{1000*(time.time()-tt)}")
    return Data


class ResultHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, shared_objects=None, **kwargs):
        self.shared_objects = shared_objects
        super().__init__(*args, **kwargs)

    def do_GET(self):
        #extract params
        if self.path.startswith( "/get_result"):
            data_id=None
            path_and_params = self.path.split('?', 1)
            path = path_and_params[0]
            params = {}
            if len(path_and_params) > 1:
                query_string = path_and_params[1]
                params = dict(qc.split("=") for qc in query_string.split("&"))
            if path == '/get_result':
                data_id = str(params.get('data_id')) # type: ignore

        #excute recognition compute task
            system = self.shared_objects['system'] # type: ignore
            id_to_ava_labels = self.shared_objects['id_to_ava_labels'] # type: ignore
            cfg = self.shared_objects['config']  # type: ignore # 存的就是cfg
            # main calculation
            tt=time.time()
            Data = process_one_clip(system, id_to_ava_labels, cfg,data_id)
            # send result back to client
            logs.info(f"🔈 processing 总耗时：{1000*(time.time()-tt)}")
            response = json.dumps(Data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response)
        else:
            self.send_error(404, "Not Found")

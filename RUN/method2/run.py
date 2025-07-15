import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

from Tools.InferenceManager import VideoBehaviorSystem
from Tools.visulize import pic_show,ava_inference_transform
from utils.camera import MyVideoCapture
from utils.Format_conversion import Yolo5LikeDetections
from utils.Data_process import build_final_data
from RUN.method2.http_result_server import launch_in_thread, update_result
from Tools.readYaml import load_config
import torch


if __name__ == "__main__":
    cfg=load_config('config.yaml')
    PERSON_CLASS_ID=cfg.get('PERSON_CLASS_ID')
    device=cfg.get('device')
    imsize=cfg.get('imsize')
    video=cfg.get('video',{})
    FPS=video.get('fps')
    slowfast=cfg.get('slowfast',{})
    alpha=slowfast.get('alpha')
    yolo=cfg.get('yolo')
    yolo_iou=yolo.get('confidence_threshold')
    yolo_conf=yolo.get('iou_threshold')
    # 使用指定的配置文件初始化系统
    system = VideoBehaviorSystem(cfg)
    #开启服务
    launch_in_thread(port=6666)

    id_to_ava_labels = {} #id means waht had processed by deepsort 
    cap = MyVideoCapture(system.input_path)
    while not cap.end:
        Data={}
        ret, img = cap.read()
        if not ret:
            continue
        res=system.yolo_model([img], imgsz=imsize, verbose=False,iou=yolo_iou,conf=yolo_conf) # type: ignore
        yolo_preds=Yolo5LikeDetections(res, [img], system.yolo_model)
        deepsort_outputs=[]
        for j in range(len(yolo_preds.pred)):
           temp=system.deepsort_update(yolo_preds.pred[j].cpu(),yolo_preds.xywh[j][:,0:4].cpu(),yolo_preds.ims[j])
           if len(temp)==0: # type: ignore
               temp=np.ones((0,8))
           deepsort_outputs.append(temp.astype(np.float32)) # type: ignore
           yolo_preds.pred=deepsort_outputs

        if len(cap.stack) == FPS:# every second
            print(f"processing {cap.idx // FPS}th second clips")
            clip = cap.get_video_clip()#25张图片拼接的矩阵
            detections = yolo_preds.pred[0]  # [n,8] -> [x1, y1, x2, y2, cls, trackid, vx, vy]
            # 根据类别过滤出 person（人）的框
            person_mask = detections[:, 4] == PERSON_CLASS_ID
            person_boxes = detections[person_mask][:, :4] 
            # 如果至少有一个人
            if person_boxes.shape[0] > 0:
                inputs, inp_boxes, _ = ava_inference_transform(clip, person_boxes, 
                                                               crop_size=imsize,num_frames=FPS,
                                                               slow_fast_alpha=alpha,device=device)
                with torch.no_grad():
                    slowfaster_preds = system.slowfast_model(inputs, inp_boxes) # type: ignore
                    slowfaster_preds = slowfaster_preds.cpu()
                for tid,avalabel in zip(detections[person_mask][:,5].tolist(), np.argmax(slowfaster_preds, axis=1).tolist()):
                    id_to_ava_labels[tid] = system.ava_labelnames[avalabel+1] # type: ignore
                    
        pic=system.save_yolopreds_tovideo(yolo_preds, id_to_ava_labels,save=False)
        Data = build_final_data(yolo_preds.pred[0], id_to_ava_labels,pic)
        update_result(Data)
        #可视化
#        pic_show(img=pic,winname='demo',fps=FPS,vis=True)


        

import cv2
import torch

def xyxy_to_xywh(xyxy: torch.Tensor) -> torch.Tensor:
    """
    将 [x1,y1,x2,y2] 转为 [cx,cy,w,h]
    xyxy: Tensor of shape [n,4]
    返回 Tensor of shape [n,4]
    """
    x1, y1, x2, y2 = xyxy.unbind(1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w  = x2 - x1
    h  = y2 - y1
    return torch.stack((cx, cy, w, h), dim=1)
# {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', ...}
   
class Yolo5LikeDetections:
    """
    将 Ultralytics YOLOv8/11 的 Results 列表转换为 YOLOv5-style 接口：
      - .pred : list of [n,6] tensors (x1,y1,x2,y2,conf,cls)
      - .pred ->list if [n.8] tensor(x1, y1, x2, y2, cls, trackid, vx, vy)
      - .xywh : list of [n,4] tensors (cx,cy,w,h)
      - .ims  : list of original image arrays (BGR numpy)
      - .names: dict idx->class name (str)
     # """
    
    def __init__(self, results, images, model):
        """
        results: list of ultralytics.engine.results.Results
        images:  iterable of the same images you传给 model(...) 的原始列表
                 —— 可以是 cv2.imread 后的 numpy.ndarray，也可以是 PIL.Image 等。
        model:   你加载的 YOLO(model.pt) 对象，用于取它的 .names
        """
        self.pred = []   # 每帧的 [x1,y1,x2,y2,conf,cls]
        self.xywh = []   # 每帧的 [cx,cy,w,h]
        self.ims  = []   # 保存原始图片 (BGR numpy array)
        # classes 映射：{0: 'person', ...}
        self.names = model.names if hasattr(model, 'names') else {}

        for res, im in zip(results, images):
            # xyxy: [n,4], conf: [n], cls: [n]
            xyxy = res.boxes.xyxy  # Tensor[n,4]
            conf = res.boxes.conf.unsqueeze(1)  # [n,1]
            cls  = res.boxes.cls.unsqueeze(1)   # [n,1]
            # 拼接成 [n,6]
            detections = torch.cat((xyxy, conf, cls), dim=1)
            self.pred.append(detections)

            # 转为 xywh 供 DeepSORT 使用
            self.xywh.append(xyxy_to_xywh(xyxy))

            # 保留原始图像（OpenCV 风格 BGR numpy）
            if isinstance(im, torch.Tensor):
                # 如果是 Tensor[C,H,W], 转回 numpy BGR
                arr = im.cpu().numpy().transpose(1,2,0)  # RGB
                arr = (arr * 255).astype('uint8')
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                self.ims.append(arr)
            else:
                # 假设已经是 BGR numpy
                self.ims.append(im)

    
    
    def __len__(self):
        return len(self.pred)

    def __getitem__(self, idx):
        """返回第 idx 帧的检测结果 Tensor[n,6]"""
        return self.pred[idx]
    
    
def serialize_yolo_preds(yolo_preds): # type: ignore
        """将 yolo_preds 转换为可 JSON 序列化的字典"""
        results = []
        for pred_tensor in yolo_preds.pred:  # pred: list of [n,6] tensors
            frame_result = []
            for det in pred_tensor.cpu().numpy().tolist():
                x1, y1, x2, y2, conf, cls_id = det
                frame_result.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": conf,
                    "class_id": int(cls_id)
                })
            results.append(frame_result)
        return {"predictions": results}
import torch
class DeepSortLikeDection:
    """
    YOLO8/YOLOv9 tracking/detection outputs封装类。
    可输入YOLO的results（单帧或多帧），
    支持以统一方式获取 pred/xywh/ims/names等，
    并自动兼容有无track_id的情况。
    """
    def __init__(self, results):
        # 支持单帧对象或者多帧list
        if not isinstance(results, list):
            results = [results]
        self.results = results
        
        self._preds = []
        self._xywhs = []
        self._ims = []
        self._names = None
        self._format_results()
        
    def _format_results(self):
        for res in self.results:
            boxes = res.boxes
            xyxy = boxes.xyxy.cpu()
            xywh = boxes.xywh.cpu()
            cls = boxes.cls.view(-1, 1).cpu()
            if hasattr(boxes, 'id') and boxes.id is not None:
                trackid = boxes.id.view(-1, 1).cpu()
            else:
                trackid = torch.full((xyxy.shape[0], 1), -1, dtype=xyxy.dtype)
            vx = torch.full((xyxy.shape[0], 1), -1, dtype=xyxy.dtype)
            vy = torch.full((xyxy.shape[0], 1), -1, dtype=xyxy.dtype)
            pred = torch.cat([xyxy, cls, trackid,vx,vy], dim=1)  # [n,6]
            
            self._preds.append(pred)
            self._xywhs.append(xywh)
            self._ims.append(res.orig_img)
            self._names = res.names

    @property
    def pred(self):
        """List of [n, 6] tensor:  (x1,y1,x2,y2,cls,trackid)"""
        return self._preds
    
    @property
    def xywh(self):
        """List of [n, 4] tensor:  (cx,cy,w,h)"""
        return self._xywhs
        
    @property
    def ims(self):
        """List of np.ndarray (BGR images)"""
        return self._ims
    
    @property
    def names(self):
        """Dict: idx->class name"""
        return self._names
    
    def __len__(self):
        return len(self.results)
        
    def __getitem__(self, idx):
        """可索引帧结果，返回dict"""
        return {
            'pred': self._preds[idx],
            'xywh': self._xywhs[idx],
            'im': self._ims[idx],
        }
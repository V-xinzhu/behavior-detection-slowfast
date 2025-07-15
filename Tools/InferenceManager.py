import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 必须放在导入torch之前
import cv2,torch,random,warnings
warnings.filterwarnings("ignore",category=UserWarning)
"""pytorchvideo"""
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
"""local YOLOv11"""
from ultralytics import YOLO # type: ignore

"""pyslowfast module"""
from pytorchvideo.models.hub import slowfast_r50_detection
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.checkpoint as cu
from slowfast.models import build_model
from slowfast.config.defaults import assert_and_infer_cfg
"""import target tracking"""
from deep_sort.deep_sort import DeepSort
from Tools.visulize import plot_one_box



class VideoBehaviorSystem:
    """
    视频行为识别系统的初始化类，加载并初始化模型、视频 I/O 等组件。
    """
    def __init__(self, cfg):
        # 加载配置文件
        self.config = cfg

        # 初始化模型和组件
        self.yolo_model       = None
        self.slowfast_model   = None
        self.UseDeepSort      = True
        self.deepsort_tracker = None
        self.ava_labelnames   = None
        self.coco_color_map   = None
        self.input_path       = 0  
        self.output_path      = None
        self.outputvideo      = None

        # 加载模型、视频 I/O、标签和颜色映射
        self.load_models()
        self.load_video_io()
        self.load_labels_and_colors()

    def load_models(self):
        """
        初始化并加载 YOLO、SlowFast、DeepSORT 等模型
        """
        # --- YOLO 模型加载 ---
        
        yolo_cfg = self.config.get('yolo', {})
        model_type = yolo_cfg.get('model_type', 'yolov5')
        model_path = yolo_cfg.get('model_path', None)
        device = self.config.get('device')
        try:
            
            warm_model = YOLO(model_path).to(device).eval() 
            with torch.no_grad():
                dummy_img = torch.zeros((1, 3, 640, 640), device=device) 
                warm_model(dummy_img)
            self.yolo_model = warm_model
            print(f"{model_type} 模型加载预热完成")
        except Exception as e:
            raise RuntimeError(f"{model_type}模型加载失败: {e}")

        # --- SlowFast 模型加载 ---
        slowfast_cfg = self.config.get('slowfast', {})
        slowfast_type = slowfast_cfg.get('model_type')
        model_cfg =slowfast_cfg.get('model_cfg')
        try:
            if model_cfg:
                args = parse_args(model_cfg)
                cfg = load_config(args, model_cfg)
                cfg = assert_and_infer_cfg(cfg)
                self.slowfast_model = build_model(cfg, gpu_id=None)
                self.slowfast_model.eval()
                cu.load_test_checkpoint(cfg, self.slowfast_model)

            else:
                # 未指定权重时加载预训练模型
                self.slowfast_model = slowfast_r50_detection(True).eval().to(device)
            if not self.slowfast_model:
                    print(f"SlowFast 模型 {slowfast_type}加载失败" )
                    exit()
            # print(f"SlowFast 模型 '{slowfast_type}' 加载成功: {slowfast_path if slowfast_path else '预训练权重'}")
        except Exception as e:
            raise RuntimeError(f"SlowFast 模型加载失败: {e}")

        # --- DeepSORT 跟踪器加载 ---
        
        deepsort_cfg = self.config.get('deepsort', {})
        deepsort_path = deepsort_cfg.get('model_path', None)
        deepsort_use = self.config.get('deepsort_use', True)
        try:
            if DeepSort is None:
                raise ImportError("DeepSORT 库未安装")
            self.deepsort_tracker = DeepSort(deepsort_path)
            print(f"DeepSORT 跟踪器加载成功: {deepsort_path}")
        except Exception as e:
            raise RuntimeError(f"DeepSORT 加载失败: {e}")

    def load_video_io(self):
        """
        初始化视频输入输出（cv2.VideoCapture 和 cv2.VideoWriter）
        """
        video_cfg = self.config.get('video', {})
        self.input_path = video_cfg.get('input', 0)   # 默认为摄像头（0）
        self.output_path = video_cfg.get('output', None)
        fps = video_cfg.get('fps')
        codec = video_cfg.get('codec')

        # 打开视频输入源
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            print(f"无法打开视频源: {self.input_path}")
        # 配置视频输出
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*codec) # type: ignore
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.outputvideo = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            print(f"视频输出已初始化: {self.outputvideo} (fps={fps}, codec={codec})")
        else:
            print("未指定视频输出，将不会写出结果视频。")
            self.outputvideo = None

    def load_labels_and_colors(self):
        """
        加载类别标签和颜色映射
        """
        labels_cfg = self.config.get('labels', {})
        label_file = labels_cfg.get('file', None)
        if label_file:
            try:
                self.ava_labelnames,_ = AvaLabeledVideoFramePaths.read_label_map(label_file)
                print(f"标签文件 '{label_file}' 加载成功，类别数量: {len(self.ava_labelnames)}")
            except Exception as e:
                print(f"无法加载标签文件: {e}")
        else:
            print("未指定标签文件。")
        self.coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]
        print("颜色配置完成。")
    
       
    def deepsort_update(self, pred, xywh, np_img):
        conf=pred[:,4:5]
        clas =pred[:,5]
        try:
            outputs = self.deepsort_tracker.update(xywh, conf,clas.tolist(),cv2.cvtColor(np_img,cv2.COLOR_BGR2RGB)) # type: ignore
        except Exception as e:
            return e
        return outputs
    
    #对图片绘制框和文本 
    def save_yolopreds_tovideo(self,yolo_preds, id_to_ava_labels,save=False):
        for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):#执行一次，因为ims只有一张图片
            im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            if pred.shape[1]==8:
                for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                    trackid = int(trackid)
                    if int(cls) != 0:
                        ava_label = ''
                    elif trackid in id_to_ava_labels.keys():#id-to-ava-labels 中已经包含 trackid
                        ava_label = id_to_ava_labels[trackid].split(' ')[0]
                    else:
                        ava_label = 'Unknown'#每32帧才会获得一个行为id,0-30帧率会赋值Unknown
                    text = '{} {} {}'.format(int(trackid),yolo_preds.names[int(cls)],ava_label)
                    color = self.coco_color_map[int(cls)] # type: ignore
                    im = plot_one_box(box,im,color,text) # type: ignore
            if save and self.outputvideo!=None:
                self.outputvideo.write(im) # type: ignore
            im = im.astype(np.uint8)
        return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)


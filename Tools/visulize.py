import cv2 
import numpy as np
import torch
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,)
from torchvision.transforms._functional_video import normalize

def pic_show(img, winname="demo", fps=25,vis=False):
    if  not vis:
        return False
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(winname, img)
    key = cv2.waitKey(int(1000 / fps))
    if key == ord('q'):
        cv2.destroyAllWindows()
        return True
    return False

def plot_one_box(box, img, color=[100,100,100], text_info="None", # type: ignore
                 velocity=None, thickness=1, fontsize=0.5, fontthickness=1):
        # Plots one bounding box on image img
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
    t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize , fontthickness+2)[0]
    cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1]*1.45)), color, -1)
    cv2.putText(img, text_info, (c1[0], c1[1]+t_size[1]+2), 
                cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255,255,255], fontthickness)
    return img

def ava_inference_transform(
clip,  # type: ignore
boxes,
num_frames = 32, #if using slowfast_r50_detection, change this to 32, 4 for slow 
crop_size = 640, 
data_mean = [0.45, 0.45, 0.45], 
data_std = [0.225, 0.225, 0.225],
slow_fast_alpha = 4, #if using slowfast_r50_detection, change this to 4, None for slow
device=None
):
    boxes = np.array(boxes)
    roi_boxes = boxes.copy()
    try:
        clip = uniform_temporal_subsample(clip, num_frames) # type: ignore
        clip = clip.float()
        clip = clip / 255.0
        # print("divided by 255, clip min:", clip.min().item(), "max:", clip.max().item())
    except  Exception as e:
        print("出错:", e)

    height, width = clip.shape[2], clip.shape[3]
    boxes = clip_boxes_to_image(boxes, height, width) # type: ignore
    clip, boxes = short_side_scale_with_boxes(clip,size=crop_size,boxes=boxes,)#分辨率缩放
    clip = normalize(clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),) 
    boxes = clip_boxes_to_image(boxes, clip.shape[2],  clip.shape[3]) # type: ignore
    if slow_fast_alpha :
        fast_pathway = clip
        slow_pathway =torch.index_select(clip,1,
            torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway, fast_pathway]
    clip = [inp.unsqueeze(0) for inp in clip]
    if device is not None:
        # Transfer the data to the current GPU device.
        if isinstance(clip, (list,)):
            for i in range(len(clip)):
                clip[i] = clip[i].cuda(
                    device=torch.device(device), non_blocking=True
                )
        else:
            clip = clip.cuda(
                device=torch.device(device), non_blocking=True
            )
    boxes=torch.from_numpy(boxes)
    boxes = torch.cat([torch.zeros(boxes.shape[0], 1), boxes], dim=1).to(device)
    
    return clip, boxes, roi_boxes

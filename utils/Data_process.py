import base64
import cv2
import torch
def imgs_bytes_list_to_clip(img_data, to_tensor):
    """
    img_data: list of image bytes (jpg)
    to_tensor: function, input为解码后的np.array,输出torch.tensor,shape [C,H,W]
    return: torch.tensor, shape [T,C,H,W]
    """
    assert len(img_data) > 0
    imgs_tensor = [to_tensor(img) for img in img_data]  # list of [C,H,W]
    # 3. 拼接: [T,C,H,W]
    # clip = torch.stack(imgs_tensor, dim=0)  # [T,C,H,W]
    clip = torch.cat(imgs_tensor).permute(-1, 0, 1, 2)
    return clip
# 示范to_tensor函数（实际可按自己的定义来）
def to_tensor(img):
    # img为HWC，转为CHW，并归一化
    img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return  img.unsqueeze(0)


def build_final_data(pred_tensor, id_to_ava_labels,img):
    """
    pred_tensor: Tensor of shape [n, 8], representing one frame's detections
    id_to_ava_labels: dict {track_id: label}
    img: 当前帧图像 (ndarray)
    """
    result = []
    detections = pred_tensor.tolist()  # 转成 list
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8') # type: ignore
    if len(detections[0])== 8:
        for det in detections:
            x1, y1, x2, y2, cls_id, track_id, vx, vy = det
            track_id = int(track_id)
            cls_id = int(cls_id)

            ava_label = id_to_ava_labels.get(track_id, "unknown")

            result.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class_id": cls_id,
                "track_id": track_id,
                "ava_label": ava_label,# 或者 base64 编码后插入 img_str
            })
    elif len(detections[0])== 6:
        for det in detections:
            x1, y1, x2, y2, cls_id, track_id = det
            track_id = int(track_id)
            cls_id = int(cls_id)

            ava_label = id_to_ava_labels.get(track_id, "unknown")

            result.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class_id": cls_id,
                "track_id": track_id,
                "ava_label": ava_label,# 或者 base64 编码后插入 img_str
            })
    return {
    "img": img_base64,
    "detections": result
}


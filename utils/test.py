import numpy as np
import requests
import time
from logger import get_logger
import cv2
import base64


logs=get_logger("module1", filename="test")

if __name__ =="__main__" :
    while True:
        a=time.time()
        try:
            res = requests.get("http://localhost:8080/get_result")
            if res.ok:
                data = res.json()
                # predictions = data
                logs.info(f"共接收到 {len(data)} 目标")
                if "img" not in data or not data["img"]:
                    logs.warning("服务端没返回图片img!")
                    continue
                img_base64 = data["img"]
                img_bytes = base64.b64decode(img_base64)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR) # type: ignore
                for i, prediction in enumerate(data["detections"]):
                    logs.info(f"\n第{i+1}个目标:")
                    x1 = prediction["x1"]
                    y1 = prediction["y1"]
                    x2 = prediction["x2"]
                    y2 = prediction["y2"]
                    cls = prediction["class_id"]
                    track_id = prediction["track_id"]
                    ava_label = prediction["ava_label"]
                    logs.info(f"目标: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], 类别: {cls}, ID: {track_id}, 行为: {ava_label}")
            img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            cv2.imshow("demo",img)
            key = cv2.waitKey(int(1000 / 25))
            if key == ord('q'):
                    break
        except Exception as e:
            logs.info("异常:", e)
            break
        time.sleep(2)
        b=time.time()
        print(f"处理时间为{b-a-2}")
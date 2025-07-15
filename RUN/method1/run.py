import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from http.server import HTTPServer
from Tools.InferenceManager import VideoBehaviorSystem
from utils.camera import MyVideoCapture
from Tools.readYaml import load_config

from functools import partial
from RUN.method1.modelServe import ResultHandler

if __name__ == "__main__":
    cfg = load_config('config.yaml')

    # 初始化系统
    system = VideoBehaviorSystem(cfg)
    # cap = MyVideoCapture(system.input_path)
    id_to_ava_labels = {}

    # 准备要共享给handler的对象
    shared_objects = {
        # 'cap': cap,
        'system': system,
        'id_to_ava_labels': id_to_ava_labels,
        'config': cfg
    }

    # 通过partial把shared_objects传入handler的构造
    HandlerClass = partial(ResultHandler, shared_objects=shared_objects)
    port=6666
    server = HTTPServer(("0.0.0.0", port), HandlerClass)
    print(f"🟢 HTTP 服务已启动：监听端口 {port}，路径 /get_result")
    server.serve_forever()
    
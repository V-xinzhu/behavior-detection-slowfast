# utils/http_result_server.py
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading

# 全局共享变量
latest_result = None
result_lock = threading.Lock()

class ResultHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/get_result":
            with result_lock:
                if latest_result is not None:
                    response = json.dumps(latest_result).encode("utf-8")
                else:
                    response = json.dumps({"status": "no data yet"}).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response)
        else:
            self.send_error(404, "Not Found")

def start_http_server(port=8080):
    server = HTTPServer(("0.0.0.0", port), ResultHandler)
    print(f" 🔊 HTTP 服务已启动：监听端口 {port}，路径 /get_result")
    server.serve_forever()

def launch_in_thread(port=8080):
    t = threading.Thread(target=start_http_server, args=(port,), daemon=True)
    t.start()

def update_result(data):
    global latest_result
    with result_lock:
        latest_result = data

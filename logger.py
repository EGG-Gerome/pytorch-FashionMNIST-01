import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir="log"):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)  # 确保目录存在
        log_index = 0
        while os.path.exists(os.path.join(log_dir, f"log{log_index}.txt")):
            log_index += 1
        self.log_path = os.path.join(log_dir, f"log{log_index}.txt")
        self.log_file = open(self.log_path, "w")

    # 封装一个带时间戳的 log 函数
    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] {msg}"
        print(full_msg)
        self.log_file.write(full_msg + "\n")
        self.log_file.flush()  # 立即写入磁盘，防止意外退出时丢失

    def close(self):
        self.log_file.close()
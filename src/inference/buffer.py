# 学习目标：只保留最新的一小段声音，让长时间录音不必无限占用内存。
# 环形缓冲区像固定数量的格子，写到末尾后回到开头，覆盖最早的数据。
# size 是最多保留多少个数；head 是下次写入的位置，count 是当前有效数量。
import threading

import numpy as np


# ======================================================================
# 1. 准备固定空间：数组存数值，锁保护并发读写步骤
# ======================================================================
class CircularBuffer:
    def __init__(self, size: int, dtype=float):
        self.size = size
        self.buffer = np.zeros(size, dtype=dtype)  # zeros 创建指定长度、初始全为 0 的 NumPy 数组。
        self.head = 0
        self.count = 0
        self.lock = threading.Lock()  # 麦克风回调写、主循环读，避免同时改动内部状态。

    # ---------- 1.1 写一个数：写到 head，随后把位置向前移 ----------
    def put(self, item):
        # with lock 在进入时加锁、退出时释放，包含发生异常的情况。
        with self.lock:
            if self.count < self.size:
                self.buffer[self.head] = item
                self.count += 1
            else:
                self.buffer[self.head] = item

            self.head = (self.head + 1) % self.size  # % 取余，让 size-1 后面回到位置 0。

    # ---------- 1.2 写一批数：先填到末尾，放不下的再从开头填 ----------
    def put_many(self, items):
        # 这里的常规调用按每批不超过 size 来理解；用切片一次复制多项，减少逐项操作。
        with self.lock:
            n = len(items)
            first_chunk = min(n, self.size - self.head)  # 本次数量与末尾剩余空间中较小的一个。
            self.buffer[self.head : self.head + first_chunk] = items[:first_chunk]
            if first_chunk < n:
                self.buffer[: n - first_chunk] = items[first_chunk:]
            self.head = (self.head + n) % self.size
            self.count = min(self.count + n, self.size)  # 有效数量最多为容量，而非累计收到的总量。

    # ---------- 1.3 读取：按“旧 → 新”顺序返回，恢复时间顺序 ----------
    def get(self):
        with self.lock:
            if self.count == 0:
                return np.array([])

            if self.count < self.size:
                return self.buffer[: self.count]  # 没写满时数据从开头排列；此切片可能共享底层数组。

            # 写满后，head 恰好指向最旧的数据；concatenate 接起尾段和首段，得到时间顺序。
            return np.concatenate((self.buffer[self.head :], self.buffer[: self.head]))


# 预测 / 验证（在单独脚本中导入 CircularBuffer）：容量为 4，依次 put 1、2、3、4、5，
# get() 应返回哪四个数？同时打印 head、count，理解为何存储顺序与返回顺序不同。
# 修改：分两次 put_many([1, 2, 3])、put_many([4, 5])，比较与逐个写入的结果。
# 自己写：先实现单数 put 和 get，再加“绕回开头”的批量写入，最后考虑加锁。

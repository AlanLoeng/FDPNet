# ------------------------------------------------------------------------
# Copyright (c) 2025 Bolun Liang(https://github.com/AlanLoeng) All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# ------------------------------------------------------------------------
import torch
import timeit

window_length = 512  # 窗口长度

def test_window_efficiency(window_func):
    """测试窗口函数生成效率的函数"""
    setup_code = f"import torch; window_length = {window_length}; window_function = {window_func}"
    stmt = "window = window_function(window_length)"
    times = timeit.repeat(stmt, setup=setup_code, repeat=10, number=1000) # 重复多次取平均
    return min(times) # 取最快的时间

window_functions = {
    "Rectangular (implicit)": "lambda length: torch.ones(length)", # 模拟矩形窗
    "Bartlett": "torch.bartlett_window",
    "Hanning": "torch.hann_window",
    "Hamming": "torch.hamming_window",
    "Blackman": "torch.blackman_window",
    "Kaiser": "torch.kaiser_window"
}

results = {}
for name, func_str in window_functions.items():
    time = test_window_efficiency(func_str)
    results[name] = time

print("窗口函数效率测试结果 (生成时间越短越高效):")
for name, time in sorted(results.items(), key=lambda item: item[1]): # 按时间排序
    print(f"{name}: {time:.6f} 秒")
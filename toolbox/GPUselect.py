import numpy as np


def pick_gpu_lowest_memory():
    try:
        import pynvml as nv
    except ImportError:
        raise ImportError("pynvml not found")

    nv.nvmlInit()
    gpu_count = nv.nvmlDeviceGetCount()
    gpu_idle_memory = []
    for i in range(gpu_count):
        handle = nv.nvmlDeviceGetHandleByIndex(i)
        try:
            memory = nv.nvmlDeviceGetMemoryInfo(handle)
            gpu_idle_memory.append(memory.total - memory.used)
        except nv.NVMLError:
            gpu_idle_memory.append(0.0)

    # return the GPU that has the largest memory left
    return np.argsort(gpu_idle_memory)[-1]

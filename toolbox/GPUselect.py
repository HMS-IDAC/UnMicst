# import subprocess, re
import numpy as np

# def pick_gpu_lowest_memory():
#     output = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE, shell=True).communicate()[0]
#     output=output.decode("ascii")
#     gpu_output = output[output.find("Memory-Usage"):]
#         # lines of the form
#         # |    0      8734    C   python                                       11705MiB |
#     memory_regex = re.compile(r"[|]\s+?\D+?.+[ ](?P<gpu_memory>\d+)MiB /")
#     rows = gpu_output.split("\n")
#     result=[]
#     for row in gpu_output.split("\n"):
#         m = memory_regex.search(row)
#         if not m:
#             continue
#         gpu_memory = int(m.group("gpu_memory"))
#         result.append(gpu_memory)
#     return np.argsort(result)[0]


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

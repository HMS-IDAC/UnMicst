import subprocess, re
import numpy as np

def pick_gpu_lowest_memory():
    output = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE, shell=True).communicate()[0]
    output=output.decode("ascii")
    gpu_output = output[output.find("Memory-Usage"):]
        # lines of the form
        # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?\D+?.+[ ](?P<gpu_memory>\d+)MiB /")
    rows = gpu_output.split("\n")
    result=[]
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_memory = int(m.group("gpu_memory"))
        result.append(gpu_memory)
    return np.argsort(result)[0]
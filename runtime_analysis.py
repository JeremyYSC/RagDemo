import os
import pprint
import time
from multiprocessing import Process, Queue

print("start")
start_time = time.time()
run_info = {}
current_state = ""

import psutil
from main import main

def get_process_mem(pid=None):
    return psutil.Process(pid).memory_info().rss / 1024 ** 2

def run_get_mem(key, pid):
    global current_state
    print("get_mem")
    while key == current_state:
        run_info[current_state]["mem"].append(get_process_mem(pid))
        time.sleep(2)

def runner(target, *args):
    global current_state
    q = Queue()
    p = Process(target=target, args=(q, *args))
    p.start()


    for flag in iter(q.get, None):
        print(flag[0])
        mem = flag[1]
        key, state = flag[0].split("_")
        if key not in run_info.keys():
            run_info[key] = {"time": [], "mem": []}
        if state == "start":
            current_state = key
            # if key != "process":
            #     mem_p = Process(target=run_get_mem, args=(key, pid))
            #     mem_p.start()
            run_info[key]["mem"].append(mem)
            run_info[key]["time"].append(time.time())
        elif state == "done":
            current_state = ""
            run_info[key]["mem"].append(mem)
            run_info[key]["time"].append(time.time())
            if key == "process":
                break

    p.join()
    # mem_p.join()

    result = {"init": {"time": (run_info["process"]["time"][0] - start_time)}}
    for k, v in run_info.items():
        result[k] = {}
        result[k]["time"] = v["time"][-1] - v["time"][0]
        result[k]["mem"] = v["mem"]
    from pprint import pprint
    pprint(result)

# def load_data(file_path):
#     from importer import import_file
#     str_mem = get_process_mem(os.getpid())
#     start = time.time()
#     import_file(file_path)
#     end_mem = get_process_mem(os.getpid())
#     end = time.time()
#     pprint.pprint(
#         {
#             "init": {
#                 "time": start - start_time,
#                 "mem": str_mem
#             },
#             "import": {
#                 "time": end - start,
#                 "mem": end_mem
#             }
#         }
#     )

if __name__ == '__main__':
    # put work here
    runner(main, "施政報告有幾頁")

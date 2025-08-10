import sys

sys.path.append("../src")
import time

from aether.call.threaded_call import ThreadTaskManager


def long_task(x, y):
    time.sleep(2)
    return x + y


manager = ThreadTaskManager()
task_id = manager.submit(long_task, 5, 3)

# 查询状态
print(manager.get_status(task_id))  # running / done

# 查询结果
import time

time.sleep(3)
print("Status:", manager.get_status(task_id))  # done
print("Result:", manager.get_result(task_id))  # 8

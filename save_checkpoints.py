import threading
import os,time

def zip_compress():
    os.system('zip -r test.zip checkpoints')

    if os.path.exists("checkpoints.zip"):
        os.remove("checkpoints.zip")
        print("已删除checkpoints.zip")
        os.system('mv test.zip checkpoints.zip')
        print("已创建checkpoints.zip")
    else:
        print(f"文件checkpoints.zip不存在")
        os.system('mv test.zip checkpoints.zip')
        print("已创建checkpoints.zip")
    os.remove("test.zip")
    print("已删除test.zip")


def task():
    print("任务执行:", time.ctime())
    # 循环执行
    threading.Timer(60, zip_compress).start() # 900

task()  # 启动定时器
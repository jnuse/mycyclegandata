import threading
import os,time

def zip_compress():
    os.system('zip -r test.zip /kaggle/working/checkpoints')

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
    zip_compress()
    # 循环执行
    threading.Timer(900, zip_compress).start()

task()  # 启动定时器
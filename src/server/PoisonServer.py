import threading

from server.NormalServer import NormalServer
from utils import ModuleFindTool


class PoisonServer(NormalServer):
    r"""
        normal server supports sync and async FL
    """
    def __init__(self, config):
        super().__init__(config)
        self.poison_accuracy_list = []
        self.poison_loss_list = []

    def run(self):
        print("Start server:")

        # 启动server中的三个线程
        self.data_getter_thread.start()
        self.scheduler_thread.start()
        self.updater_thread.start()

        self.scheduler_thread.join()
        print("scheduler_thread joined")
        self.updater_thread.join()
        print("updater_thread joined")
        self.data_getter_thread.kill()
        self.data_getter_thread.join()
        print("data_getter_thread joined")

        # 队列报告
        self.queue_manager.stop()
        self.accuracy_list, self.loss_list = self.updater_thread.get_accuracy_and_loss_list()
        self.poison_accuracy_list, self.poison_loss_list = self.updater_thread.get_poison_accuracy_and_loss_list()
        # 结束主类
        self.kill_main_class()
        print("End!")
        
        
    def get_poison_accuracy_and_loss_list(self):
        return self.poison_accuracy_list, self.poison_loss_list
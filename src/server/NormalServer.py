import threading

from server.BaseServer import BaseServer
from utils import ModuleFindTool


class NormalServer(BaseServer):
    r"""
        normal server supports sync and async FL
    """
    def __init__(self, config):
        super().__init__(config)

        self.mutex_sem = threading.Semaphore(1)
        self.empty_sem = threading.Semaphore(1)
        self.full_sem = threading.Semaphore(0)

        # initialization of the server
        # the process has an order
        queue_manager_class = ModuleFindTool.find_class_by_path(self.queue_manager_config['path'])
        self.queue_manager = queue_manager_class(self.queue_manager_config)
        self.global_var['queue_manager'] = self.queue_manager

        scheduler_class = ModuleFindTool.find_class_by_path(self.server_config['scheduler']['path'])
        self.scheduler_thread = scheduler_class(self.server_thread_lock, self.server_config["scheduler"],
                                                self.mutex_sem, self.empty_sem, self.full_sem)
        self.global_var['scheduler'] = self.scheduler_thread

        updater_class = ModuleFindTool.find_class_by_path(self.server_config['updater']['path'])
        self.updater_thread = updater_class(self.server_thread_lock, self.stop_event, self.server_config['updater'],
                                            self.mutex_sem, self.empty_sem, self.full_sem)
        self.global_var['updater'] = self.updater_thread

    def kill_main_class(self):
        del self.scheduler_thread
        del self.updater_thread
        del self.queue_manager
        self.mutex_sem.release()
        self.empty_sem.release()
        self.full_sem.release()

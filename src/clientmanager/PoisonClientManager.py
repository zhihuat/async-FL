from clientmanager.NormalClientManager import NormalClientManager
from core.Runtime import CLIENT_STATUS
from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter
from core.MessageQueue import EventFactory


class PoisonClientManager(NormalClientManager):
    def __init__(self, whole_config):
        super().__init__(whole_config)
        self.global_var = GlobalVarGetter.get()
        self.client_list = []  # client list
        self.client_id_list = []  # client id list
        self.client_status = []  # client status list

        self.multi_gpu = whole_config["global"]["multi_gpu"]
        self.total_client_num = whole_config["global"]["client_num"]
        self.client_num = whole_config["client_manager"]["init_client_num"] if "init_client_num" in whole_config[
            "client_manager"] else self.total_client_num
        
        self.clean_client_num = whole_config["global"]["clean_client_num"]
        self.poison_client_num = whole_config["global"]["poison_client_num"]
        assert self.poison_client_num + self.clean_client_num == self.total_client_num
        
        self.client_staleness_list = whole_config["client_manager"]["stale_list"]
        self.index_list = whole_config["client_manager"]["index_list"]  # each client's index list
        self.client_config = whole_config["client"]
        self.poison_client_config = whole_config["poison_client"]

        self.client_dev = self.get_client_dev_list(self.total_client_num, self.multi_gpu)
        self.client_class = ModuleFindTool.find_class_by_path(whole_config["client"]["path"])
        self.poison_clent_class = ModuleFindTool.find_class_by_path(whole_config["poison_client"]["path"])
        
        self.stop_event_list = [EventFactory.create_Event() for _ in range(self.client_num)]
        self.selected_event_list = [EventFactory.create_Event() for _ in range(self.client_num)]
        self.global_var['selected_event_list'] = self.selected_event_list

    def start_all_clients(self):
        self.__init_clients()
        # start clients
        self.global_var['client_list'] = self.client_list
        self.global_var['client_id_list'] = self.client_id_list
        print("Starting clients")
        for i in self.client_id_list:
            self.client_list[i].start()
            self.client_status[i] = CLIENT_STATUS['active']

    def __init_clients(self):
        for i in range(self.poison_client_num):
            self.client_list.append(
                self.poison_clent_class(i, self.stop_event_list[i], self.selected_event_list[i], self.client_staleness_list[i],
                                  self.index_list[i], self.poison_client_config, self.client_dev[i]))  # instance
            self.client_status.append(CLIENT_STATUS['created'])
            self.client_id_list.append(i)
        
        for i in range(self.poison_client_num, self.clean_client_num+self.poison_client_num):
            self.client_list.append(
                self.client_class(i, self.stop_event_list[i], self.selected_event_list[i], self.client_staleness_list[i],
                                  self.index_list[i], self.client_config, self.client_dev[i]))  # instance
            self.client_status.append(CLIENT_STATUS['created'])
            self.client_id_list.append(i)


from dpwa.conn import RxThread, TxThread
import yaml
import time
from queue import Queue

import logging
LOGGER = logging.getLogger(__name__)

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __repr__(self):
        return 'Struct: ' + repr(self.__dict__)

class DpwaConfiguration:
    def __init__(self, config_file):
        self.yaml = yaml.load(open(config_file, 'rt'),Loader=yaml.FullLoader)
        self.config = {}
        for c in self.yaml:
            k = list(c.keys())[0]
            self.config[k] = c[k]

    def get_nodes(self):
        return self.config['nodes']



class DpwaControl:
    def __init__(self, name,config_file):  
        self.name = name
        # The clock is used to keep track of the model's age in terms of
        # training samples trained so far (increase by 1 in update_send())
        self.clock = 0
        self.loss=0
        self.fetching = False

        self._queue_from_son = Queue()
        self._queue_from_father = Queue()

        self.config = DpwaConfiguration(config_file)
        self.allnodes = self.config.get_nodes()

        # allnodes 结构
        # [
        #     {'name': 'w0', 'host': 'localhost', 'port': 45000, 'father': None, 'son': 'w1'}
        #     {'name': 'w1', 'host': 'localhost', 'port': 45001, 'father': 'w0', 'son': ['w2', 'w3']}
        #     {'name': 'w2', 'host': 'localhost', 'port': 45002, 'father': 'w1', 'son': None}
        #     {'name': 'w3', 'host': 'localhost', 'port': 45003, 'father': 'w1', 'son': None}
        # ]

        # 初始化所有的节点信息
        self.peers = []  
        for node in self.allnodes:
            node = Struct(**node)   # 将dict形式的node转换成一个实例对象，使用update()方法将dict所有的键值转换成对象成员,进而使用.的形式访问各个属性
            if node.name == name:
                self.me=node
            else:
                self.peers.append(node)

  
        self.rx = RxThread(self.me.name,self.me.host, self.me.port)  #, timeout_ms的参数之后再做考虑
        self.tx = TxThread(self.me.name)
        

        self.fathers=[]
        self.sons=[]
        for peer in self.peers:
            if peer.name in self.me.fathers:
                self.fathers.append(peer)
                continue
            if peer.name in self.me.sons:
                self.sons.append(peer)

        for father in self.fathers:
            self.add_father(father.name,father.host,father.port)

            # 每有一个father, rx线程就需要接受一个model.data
            self._queue_from_father.put(True)

        for son in self.sons:
            self.add_son(son.name,son.host,son.port)

            # 每有一个son, rx线程就需要接受一个model.data
            self._queue_from_son.put(True)

        # print('fathers...')
        # self.tx.showfathers()
        # print('son...')
        # self.tx.showsons()

        # time.sleep(100)
        # input()
        self.rx.start()
        self.tx.start()

    def add_father(self, name, host, port):
        self.tx.add_father(name, host, port)
    
    def add_son(self, name, host, port):
        self.tx.add_son(name, host, port)
    

    def sendFather(self, parameters): #parameters为训练网络的参数
        """Initiate an update to the cluster.

        Performs 2 things:
        1. Updates the local server with the latest parameters, so other peers could fetch them
        2. Initiate a fetch parameters request to a random peer.
        """
        # Increase the clock value
        self.clock += 1
        self.loss+=0.25
        # Serve the new parameters
        state = {'clock': self.clock, 'loss': self.loss, 'source':self.name}
        # 在rx线程中保存此时的loss和模型参数等
        # self.rx.set_current_state(state, parameters)

        self.fetching = True

        self.tx.sendFather(state,parameters)



    def gatherFromSon(self):
        """

            确保所有儿子节点的数据都已经接收到了
        """
        parametersFromSons=[]
        statesFromSons=[]
        baseLoss=self.loss
        while not self._queue_from_son.empty():
            peer_state, peer_parameters = self.rx.getFromSon()

            LOGGER.debug('%s receive from son %s loss = %s' %(self.name,peer_state['source'],peer_state['loss']))
            self.loss+=peer_state['loss']-baseLoss

            self._queue_from_son.get()
            self._queue_from_son.task_done()
            parametersFromSons.append(peer_parameters)
            statesFromSons.append(peer_state)
        LOGGER.debug('%s gather all sons, after add ... loss = %s' %(self.name,self.loss))
        self._queue_from_son.join()

        # 恢复队列, 进行下一次的数据接收
        for son in self.sons:
            self._queue_from_son.put(True)

        return parametersFromSons,statesFromSons

    def sendSon(self, parameters): #parameters为训练网络的参数


        self.clock += 1

        # Serve the new parameters
        state = {'clock': self.clock, 'loss': self.loss, 'source':self.name}
        # 在rx线程中保存此时的loss和模型参数等
        # self.rx.set_current_state(state, parameters)

        self.tx.sendSon(state,parameters)

    def updateFromFather(self):
        """

            确保所有父亲节点的数据都已经接收到了
        """

        parametersFromFathers=[]
        statesFromFathers=[]

        if not self.fetching:
            return parametersFromFathers,statesFromFathers

        while not self._queue_from_father.empty():
            peer_state, peer_parameters = self.rx.getFromFather()
            LOGGER.debug('%s receive from father %s loss = %s' %(self.name,peer_state['source'],peer_state['loss']))
            self.loss=peer_state['loss']

            self._queue_from_father.get()
            self._queue_from_father.task_done()
            parametersFromFathers.append(peer_parameters)
            statesFromFathers.append(peer_state)
        
        LOGGER.debug('%s update all fathers  ,after replace ... loss = %s' %(self.name,self.loss))
        self._queue_from_father.join()

        # 恢复队列, 进行下一次的数据接收
        for father in self.fathers:
            self._queue_from_father.put(True)

        self.fetching =False
        return parametersFromFathers,statesFromFathers


    def shutdown(self):
        self.rx.shutdown()
        self.tx.shutdown()

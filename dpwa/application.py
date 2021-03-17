from dpwa.control import DpwaControl
import pickle
import torch
import numpy as np

import logging
LOGGER = logging.getLogger(__name__)


TYPE_CONVERSION = {
    'torch.cuda.FloatTensor': np.float32,
    'torch.FloatTensor': np.float32
}

def _tensor_to_buffer(t):
    return bytes(t.numpy())

def _tensor_from_buffer_like(buf, t):
    n = np.frombuffer(buf, dtype=TYPE_CONVERSION[t.type()])
    result = torch.from_numpy(n).view(t.size())

    return result

def _serialize_bytes_dict(params):
    return pickle.dumps(params)


def _deserialize_bytes_dict(blob):
    return pickle.loads(blob)




class DpwaApplication:
    def __init__(self, net, name,config_file):  
        self.name=name      
        self._net = net
        self._conn = DpwaControl(name,config_file)       #, config_file)

    def sendFather(self):
        """

            本地训练完成之后向父节点发送模型数据
        """

        self.gatherFromSon()  # 首先应该聚合本地数据和儿子节点的数据

        print(self.name)
        params = {}
        for name, param in self._net.named_parameters():
            params[name] = _tensor_to_buffer(param.data)
        blob = _serialize_bytes_dict(params)
        #blob为此时训练网络的参数
        self._conn.sendFather(blob)

    def gatherFromSon(self):
        """
            将本地节点的数据和来自儿子节点的数据进行聚合

        """


        blobs,states= self._conn.gatherFromSon()  # states暂时没有用处

        if len(blobs)==0:    # 叶子节点的情况
            return

        
        other_params=[]
        for son_param in blobs:
            other_params.append(_deserialize_bytes_dict(son_param))

        
        for name, param in self._net.named_parameters():
            node_num=0
            for SonParam in other_params:
                t = _tensor_from_buffer_like(SonParam[name], param.data)
                param.data = t + param.data
                node_num=node_num+1
            node_num=node_num+1  # 要加上自己
            param.data = param.data/float(node_num)
        
        LOGGER.info('%s has finished the model merge' %(self.name))

    def sendSon(self):
        """

            本地更新模型并,向子节点发送模型数据
        """
        
        self.updateFromFather()

        params = {}
        for name, param in self._net.named_parameters():
            params[name] = _tensor_to_buffer(param.data)
        blob = _serialize_bytes_dict(params)
        #blob为此时训练网络的参数
        self._conn.sendSon(blob)

    def updateFromFather(self):
        """
            获取来自父节点的数据, 并直接取代本地的模型
        """


        blobs,states= self._conn.updateFromFather()
        if len(blobs)==0:
            return

        # 此处只有一个父节点    
        blob=blobs[0]
        other_params = _deserialize_bytes_dict(blob)
        for name, param in self._net.named_parameters():
            t = _tensor_from_buffer_like(other_params[name], param.data)
            param.data = t
        
        LOGGER.info('%s has finished the model replace' %(self.name))

    def shutdown(self):
        self._conn.shutdown()




if __name__ == '__main__':
    DpwaApp=DpwaApplication('net','wry')
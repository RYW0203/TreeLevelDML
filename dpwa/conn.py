import socketserver
import socket
from copy import deepcopy
import threading
from threading import Thread, Lock
from dpwa.messaging import send_message, recv_message
from queue import Queue
import time


import logging
LOGGER = logging.getLogger(__name__)

MODEL_GATHER=1
MODEL_BROADCAST=2

TCP_SOCKET_BUFFER_SIZE = 8 * 1024 * 1024



rx_queue_from_son = Queue()

rx_queue_from_father = Queue()

servername=''

class Myserver(socketserver.BaseRequestHandler):
    
    def handle(self):

        global servername
        global rx_queue_from_son

        global rx_queue_from_father

        client_sock=self.request
        while True:
          try:
            # The socket is blocking
            # LOGGER.debug("RxThread: receiving message fd=%d", client_sock.fileno())
            message_type, rev_state, rev_payload = recv_message(client_sock)
            
            rev_msg = {}
            rev_msg['msg'] = rev_state
            rev_msg['payload'] = rev_payload

            if message_type == MODEL_GATHER:
                send_message(client_sock, MODEL_GATHER) 
                rx_queue_from_son.put(rev_msg)
                LOGGER.info('server : %s has received from son %s  loss = %s ' %(servername,rev_state['source'],rev_state['loss']))
            
            if message_type == MODEL_BROADCAST:
                send_message(client_sock, MODEL_BROADCAST)  
                rx_queue_from_father.put(rev_msg)
                LOGGER.info('server : %s has received from father %s  loss = %s ' %(servername,rev_state['source'],rev_state['loss']))
                            
          except (BrokenPipeError, ConnectionResetError):
            LOGGER.warning("Other end had a timeout, socket closed")
            client_sock.close()
            break

          except:
            LOGGER.exception("Error handling request (closing socket, client will retry)")
            client_sock.close()
            break      


class RxThread(Thread):
    def __init__(self, name, bind_host, bind_port):
        super(RxThread, self).__init__()
        LOGGER.info("Starting RxThread. listening on %s:%d...", bind_host, bind_port)
        global servername
        servername=name
        self.bind_host = bind_host
        self.bind_port = bind_port

        
    def getFromSon(self):
        global servername
        global rx_queue_from_son

        witem = rx_queue_from_son.get(block=True)
        self.peer_message = witem['msg']
        self.peer_payload = witem['payload']
        rx_queue_from_son.task_done()
        return self.peer_message, self.peer_payload


    def getFromFather(self):

        global rx_queue_from_father

        witem = rx_queue_from_father.get(block=True)
        self.peer_message = witem['msg']
        self.peer_payload = witem['payload']
        rx_queue_from_father.task_done()
        return self.peer_message, self.peer_payload

    def run(self):
        print("RxThread: run()")
        # input('输入回车继续')
        try:
            # while True:
                # server = socketserver.ThreadingTCPServer((self.bind_host,self.bind_port),Myserver)
            server=socketserver.ThreadingTCPServer((self.bind_host,self.bind_port),Myserver)
            print ("socket server start.....")
            # time.sleep(2)
            server.serve_forever()
            print('sever has been shutdowned')
        except Exception as e:
               print('===>Rxthead exception:',self.bind_host,self.bind_port)
               print(str(e))

        

    def shutdown(self):
        # TODO(guyz): Implement using eventfd...
        # raise NotImplementedError
        self.server.shutdown()


def _create_tcp_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, TCP_SOCKET_BUFFER_SIZE)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, TCP_SOCKET_BUFFER_SIZE)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


class WorkerConn:
    """Describes a peer's state in the cluster."""
    def __init__(self, name, host, port):
        self.name = name
        self.host = host
        self.port = port
        # Lazy connection
        self.connected = False
        self.sock = None
    
    def __repr__(self):
        return repr(self.__dict__)


class TxThread(Thread):
    def __init__(self, name):   # name参数暂时无作用
        super(TxThread, self).__init__()

        self.sock=None
        self.peer_payload = None
        self.peer_message = None
        self.name=name
        self._queue=Queue()

        self.fathers={}
        self.sons={}
        self.fathers_lock = Lock()
        self.sons_lock=Lock()

    def add_father(self, name, host, port):
        conn = WorkerConn(name, host, port)
        with self.fathers_lock:
            self.fathers[name]=conn
    
    def add_son(self, name, host, port):
        conn = WorkerConn(name, host, port)
        with self.sons_lock:
            self.sons[name]=conn
    
    def showfathers(self):
        for f in self.fathers:
            print(f)
            print(self.fathers[f])
        
    def showsons(self):
        for s in self.sons:
            print(s)
            print(self.sons[s])


    def _get_peer(self,peer_name):
       
        if peer_name  in self.fathers.keys():
            peer=self.fathers[peer_name]
        if peer_name in self.sons.keys():
            peer=self.sons[peer_name]

     
        # 未考虑异常情况
        # self.selected_peer = peer
        # LOGGER.debug("peer %s is selected",peer.name)
             

        # Make sure the client is connected
        try:
            if not peer.connected:
                peer.sock = _create_tcp_socket()
                # peer.sock.settimeout(self.socket_timeout_ms/1000)
                peer.sock.connect((peer.host, peer.port))
                peer.connected = True
                # LOGGER.debug("connected to peer %s successfully", peer.name)
        except ConnectionRefusedError:
            LOGGER.debug("peer %s not listening yet", peer.name)
            return None
        except:
            LOGGER.exception("Couldn't connect to peer %s (unrecoverable)" %peer.name)
            # self.remove_peer(peer.name)
            #reduce the bw of the unreachable peer
            return None

        return peer    


    def run(self):
        LOGGER.info("TxThread: run()")

        while True:
            witem = self._queue.get(block=True)   #如果队列中没有值可以返回，就会被阻塞在这里
            if not witem:
                print("Exiting TxThread...")
                break

            # Wait until we succefully fetch from a peer,
            # or until we don't have any peers to fetch from
            done = False
            while not done:
                peer = self._get_peer(witem['recv_node'])
                
                try:
                    # Send a fetch parameters request
                    
                    send_message(peer.sock, witem['msgtype'], witem['msg'], witem['payload'])
                    message_type, _, _ = recv_message(peer.sock)
                    assert message_type == witem['msgtype']
                    done=True
                    LOGGER.debug('%s send model to %s %s , myloss = %s' %(self.name,witem['type'],witem['recv_node'],witem['msg']['loss']))

                except socket.timeout:
                    LOGGER.warning("TxThread: peer %s timeout, restarting connection...", peer.name)
                    peer.sock.close()
                    peer.sock = None
                    peer.connected = False
      

                except:
                    LOGGER.exception("Error connecting with peer %s.", peer.name)
                    peer.sock.close()
                    peer.sock = None
                    peer.connected = False
                    #self.remove_peer(peer.name)



            self._queue.task_done()        # 由于没有join(), 个人认为这句话可以去掉

        print("TxThread: exiting...")
        self._queue.task_done()

    def sendFather(self,state,parameter):
        
        # 有几个父亲就发送在sendToFather中存入几个模型参数
        # 如果是根节点, 由于self.fathers为空, 所以不会进入run 函数发送数据
        name=''
        for i in self.fathers:   # self.fathers的形式为字典{name:WorkerConn, name:WorkerConn, name:WorkerConn}
            name=i
            item={}
            item['type']='father'
            item['msgtype']= MODEL_GATHER
            item['recv_node'] = name
            item['msg'] = state
            item['payload'] = parameter
            self._queue.put(item)

    def sendSon(self,state,parameter):
        
        # 有几个儿子就发送在_queuer中存入几个模型参数
        # 如果是叶子节点, 由于self._queues为空, 所以不会进入run 函数发送数据
        name=''
        for i in self.sons:   # self.fathers的形式为字典{name:WorkerConn, name:WorkerConn, name:WorkerConn}
            name=i
            item={}
            item['type']='son'
            item['msgtype']= MODEL_BROADCAST
            item['recv_node'] = name
            item['msg'] = state
            item['payload'] = parameter
            self._queue.put(item)

    # def fetch_wait(self):
    #     """Waits for the fetch_parameters request to complete."""
    #     self._queue.join()
    #     return self.peer_message, self.peer_payload

    def shutdown(self):
        self._queue.put(False)
        print('shutdown join ... ')
        # self.join()
        print('-----------------------')
        self._queue.join()
        print('-----------------------')
        # print('shutdown join ... ')
        # self.join()

import numpy as np
import socket


class RobotAds:
    """
    通过机器人ADS获取参数。
    :param server_ip: 服务端IP地址.
    :param server_port: 服务端端口.

    getActualAngle:     获取关节角度

    getEE:              获取基坐标系末端位姿

    getEeWork:          获取工件坐标系末端位姿

    返回numpy.ndarray float64 shape=[6,]
    """

    def __init__(self, server_ip, server_port):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((server_ip, server_port))

    def __del__(self):
        self.client_socket.close()

    def receive_all_data(self, expected_bytes):
        """接收指定字节数的数据"""
        data = b''
        while len(data) < expected_bytes:
            remaining = expected_bytes - len(data)
            new_data = self.client_socket.recv(remaining)
            if not new_data:
                # 如果没有接收到新数据，可能连接已经关闭或发生错误
                raise Exception("Connection closed or error occurred while receiving data")
            data += new_data
        return data

    def getActualAngle(self) -> np.ndarray:
        # 发送数据
        message = "a\n"
        self.client_socket.send(message.encode())
        ActualAngle_Buff = self.receive_all_data(6 * 8)
        return np.frombuffer(ActualAngle_Buff, dtype=np.double)

    def getEE(self) -> np.ndarray:
        message = "e\n"
        self.client_socket.send(message.encode())
        EE_Buff = self.receive_all_data(6 * 8)
        return np.frombuffer(EE_Buff, dtype=np.double)

    def getEeWork(self) -> np.ndarray:
        message = "ew\n"
        self.client_socket.send(message.encode())
        EW_Buff = self.receive_all_data(6 * 8)
        return np.frombuffer(EW_Buff, dtype=np.double)


def get_All(ads_client: RobotAds):
    actualAngle = ads_client.getActualAngle()
    ee = ads_client.getEE()
    ew = ads_client.getEeWork()
    print('actual angle', actualAngle)
    print('ee', ee)
    print('ee_work', ew)


def get_AllData_in_while(ads_client: RobotAds):
    counter = 0
    while True:
        actualAngle = ads_client.getActualAngle()
        ee = ads_client.getEE()
        ew = ads_client.getEeWork()
        if counter % 100 == 0:
            print(f"counter: {counter}")
        counter += 1


if __name__ == '__main__':
    # 服务器IP和端口
    server_ip = '192.168.71.60'
    # server_ip = 'localhost'
    server_port = 8234
    ads = RobotAds(server_ip, server_port)

    get_All(ads)
    # get_AllData_in_while(ads)

import socket
import math
import time
from enum import Enum
import numpy as np

class TcpCommandObject(Enum):
    # AGV = 1
    JKROBOT = 2
    # GRASP = 3
    # ROBOTS = 3
#
class TcpCommandType(Enum):
    # ENABLE = 1
    # DISABLE = 2
    JOINTMOVE = 3
    ENDVELMOVE = 4
    # ENDRELMOVE = 5
    ENDMOVE = 6
    ENDMOVECIRCLE = 20
    ENDMOVEIMMEDIATE = 21

    # NAVLOCATION = 7
    # NAVSTATION = 8
    # FORWARD = 9
    # BACK = 10
    # LEFT = 11
    # RIGHT = 12

    IOCONTROL = 13
    SUCKSTART = 14
    SUCKSTOP = 19

    STOP = 15

    GETENDPOS = 16
    GETENDVEL = 17
    GETJOINTPOS = 18



class RobotTcp:
    def __init__(self, ip, port):
        # 通过设置标志位 是否通过打印信息调试
        self.debug = True
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip, port))
        print("robot tcp 对象被创建与连接：\nip:{0}\nport:{1}".format(self.ip, self.port))

    def myprint(self, str):
        if self.debug:
            print(str)

    # 机械臂
    # def enable_jkrobot(self):
    #     rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.ENABLE, " ",0)
    #     self.myprint(rec_data)

    # def disable_jkrobot(self):
    #     rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.DISABLE, " ", 0)
    #     self.myprint(rec_data)

    def stop_jkrobot(self):
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.STOP, " ", 1)
        self.myprint(rec_data)

    # def stop_robots(self):
    #     rec_data = self.send(TcpCommandObject.ROBOTS, TcpCommandType.STOP, " ", 1)
    #     self.myprint(rec_data)

    # eg: joints_angle = [-90.0 (°),0.0,-90.0,0.0,90.0,90.0]
    def joints_move_jkrobot(self, joints_angle):
        data = self.list2str(joints_angle)
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.JOINTMOVE, data, 0)
        self.myprint(rec_data)

    # # eg: end_relpos = [0.0 (m),0.0,0.0,0.0,0.0,90.0 (°)]
    # def end_relmove_jkrobot(self, end_relpos):
    #     data = self.list2str(end_relpos)
    #     rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.ENDRELMOVE, data, 0)
    #     self.myprint(rec_data)

    # eg: end_vel = [0.0 (cm/s),0.0,0.0,0.0,0.0,5.0 (°/s)]
    def end_vel_jkrobot(self, end_vel):
        data = self.list2str(end_vel)
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.ENDVELMOVE, data, 1)
        self.myprint(rec_data)

    # eg: end_abspos = [? (m), ?, ?, ?, ?, ? (°)]
    def end_absmove_jkrobot(self, end_abspos):
        data = self.list2str(end_abspos)
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.ENDMOVE, data, 0)
        self.myprint(rec_data)

    # eg: p_end = [? (m), ?, ?, ?, ?, ? (°)]
    def end_absmovecircle_jkrobot(self, p_len, radius, p_transition, p_end, vel=0.01, acc=0.5, dec=0.5):
        circle_pos = []
        circle_pos.append(p_len)
        circle_pos.append(radius)
        circle_pos += p_transition + p_end
        circle_pos.append(vel)
        circle_pos.append(acc)
        circle_pos.append(dec)
        data = self.list2str(circle_pos)
        print(data)
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.ENDMOVECIRCLE, data, 0)
        self.myprint(rec_data)

    # eg: end_pos = [? (m), ?, ?, ?, ?, ? (°)]
    def end_absmoveimmediate_jkrobot(self, end_pos):
        data = self.list2str(end_pos)
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.ENDMOVEIMMEDIATE, data, 1)
        self.myprint(rec_data)

    # eg: return [x (m), y, z, r (deg), p, y]
    def get_end_pos_jkrobot(self):
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.GETENDPOS, " ", 1)
        # self.myprint(rec_data)
        datas = rec_data[1:-1].split(",")
        output = [float(data) for data in datas]
        return output

    # eg: return [vx (cm/s), vy, vz, vr (deg/s), vp, vy]
    def get_end_vel_jkrobot(self):
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.GETENDVEL, " ", 1)
        # self.myprint(rec_data)
        datas = rec_data[1:-1].split(",")
        output = [float(data) for data in datas]
        return output

    # eg: return [j1 (deg), j2, j3, j4, j5, j6]
    def get_joint_pos_jkrobot(self):
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.GETJOINTPOS, " ", 1)
        # self.myprint(rec_data)
        datas = rec_data[1:-1].split(",")
        output = [float(data) for data in datas]
        return output


    # SUCK
    # eg: set_io_id = [2, 4], reset_io_id = [2, 4]
    def io_control(self, set_io_id: list = [], reset_io_id: list = []):
        # 一共16个IO口可供控制，值为1则置1，值为-1则置0，值为0则保持不变
        control_list = []
        for i in range(16):
            if i in set_io_id:
                control_list.append(1)
            elif i in reset_io_id:
                control_list.append(-1)
            else:
                control_list.append(0)
        data = self.list2str(control_list)
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.IOCONTROL, data, 1)
        self.myprint(rec_data)

    # eg: suck_io_id = [2, 4]
    def suck_start(self, suck_io_id):
        self.io_control(set_io_id=suck_io_id)

    # eg: suck_io_id = [2, 4] unsuck_io_id = [3, 5] 吸真空与破真空IO口号可以不对应
    def suck_stop(self, suck_io_id, unsuck_io_id):
        self.io_control(reset_io_id=suck_io_id)
        self.io_control(set_io_id=unsuck_io_id)
        time.sleep(0.1)
        self.io_control(reset_io_id=unsuck_io_id)

    # # AGV
    # # eg: station_id = 1
    # def agv_nav_station(self, station_id):
    #     rec_data = self.send(TcpCommandObject.AGV, TcpCommandType.NAVSTATION, str(station_id), 0)
    #     self.myprint(rec_data)
    #
    # # eg: agv_posture = [-0.1 (m), 0.1, 45 (°)]
    # def agv_nav_location(self, agv_posture):
    #     x = round(1000 * agv_posture[0])
    #     y = round(1000 * agv_posture[1])
    #     yaw = round(1000 * agv_posture[2] * 2 * math.pi / 360)
    #     int16_x_low, int16_x_high = ui32toui16(x)
    #     int16_y_low, int16_y_high = ui32toui16(y)
    #     int16_yaw_low, int16_yaw_high = ui32toui16(yaw)
    #     data = str(int16_x_high) + "," + str(int16_x_low) + "," + str(int16_y_high) + "," + str(int16_y_low)  + "," + str(int16_yaw_high)  + "," + str(int16_yaw_low)
    #     rec_data = self.send(TcpCommandObject.AGV, TcpCommandType.NAVLOCATION, data, 0)
    #     self.myprint(rec_data)
    #
    # def agv_forward(self):
    #     rec_data = self.send(TcpCommandObject.AGV, TcpCommandType.FORWARD, " ", 0)
    #     self.myprint(rec_data)
    #
    # def agv_back(self):
    #     rec_data = self.send(TcpCommandObject.AGV, TcpCommandType.BACK, " ", 0)
    #     self.myprint(rec_data)
    #
    # def agv_left(self):
    #     rec_data = self.send(TcpCommandObject.AGV, TcpCommandType.LEFT, " ", 0)
    #     self.myprint(rec_data)
    #
    # def agv_right(self):
    #     rec_data = self.send(TcpCommandObject.AGV, TcpCommandType.RIGHT, " ", 0)
    #     self.myprint(rec_data)
    def list2str(self, data):
        return str(data).replace("[", "").replace("]", "")

    # object: 控制对象 command: 控制命令 data: 控制数据 emergency: 是否紧急指令1 OR 0
    def send(self, object, command, data, emergency):
        msg = bytes(object.name + "#" + command.name + "#" + data + "#" + str(emergency), encoding='utf-8')
        self.sock.sendall(msg)
        data = self.sock.recv(1024)
        return data.decode()

    def close(self):
        self.sock.close()

    def __enter__(self):
        print("robot tcp 对象进入with：\nip:{0}\nport:{1}".format(self.ip, self.port))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        print("robot tcp 对象退出with：\nip:{0}\nport:{1}".format(self.ip, self.port))


    def __del__(self):
        self.close()
        print("robot tcp 对象被析构：\nip:{0}\nport:{1}".format(self.ip, self.port))


def ui32toui16(input):
    int32_num = input
    int16_num_low = int32_num & 0xFFFF
    int16_num_high = (int32_num >> 16) & 0xFFFF
    return (int16_num_low, int16_num_high)

if __name__ == '__main__':
    ############## enter、exit 函数测试 ##############
    # with RobotTcp('192.168.71.60', 8001) as robot_tcp:
    #     pass
    ############## del 函数测试 ##############
    # robot_tcp = RobotTcp('192.168.71.60', 8001)
    robot_tcp = RobotTcp('172.16.11.132', 8001)
    ############## JKRobot 函数测试 ##############
# 关节角度 deg
    # robot_tcp.joints_move_jkrobot([-10.464647, -3.093725, 116.709981, 67.138663, 79.326782, -89.999409])
# 末端位置 m, deg
#
    # while True:

        # robot_tcp.end_absmove_jkrobot([-0.735, 0.021, 0.301, 9.0e+01, 0.0, -9.0e+01]) # 开门初始动作
        # robot_tcp.end_absmove_jkrobot([-0.735, 0.021, 0.891, 9.0e+01, 0.0, -9.0e+01]) # 开门结束动作
        # robot_tcp.end_absmove_jkrobot([-0.615, -0.081, 0.712, 9.0e+01, 0.0, -9.0e+01]) # 过度视野中心动作
    # robot_tcp.end_absmove_jkrobot([-0.946, -0.059, 0.623, 9.0e+01, 0.0, -9.0e+01]) # 视野中心动作
    robot_tcp.end_absmove_jkrobot([-0.442576, 0.00507468, 0.38258, 9.0e+01, 0.0, -9.0e+01])  # 初始位置


    # robot_tcp.end_absmove_jkrobot([-0.562863, 0.00507468, 0.721458, 55.155, 0.0, -9.0e+01]) # 底座检测位姿
    # robot_tcp.end_absmove_jkrobot([-0.442576, 0.00507468, 0.553, 9.0e+01, 0.0, -9.0e+01]) # 上升到底座高度之上
    # robot_tcp.end_absmove_jkrobot([-1.13846, 0.0213097, 0.553, 9.0e+01, 0.0, -9.0e+01]) # 移动到顶座位置上面11cm
    # robot_tcp.end_absmove_jkrobot([-1.13846, 0.0213097, 0.44329, 9.0e+01, 0.0, -9.0e+01]) # 移动到顶座位置
# 末端位置走过渡圆弧
#     robot_tcp.end_absmovecircle_jkrobot([-0.45501907, -0.0458132, 0.10520692, 8.97304176e+01, -5.52288735e-03, -9.00018932e+01],
#                                         0.1, 0.1)
#     robot_tcp.end_absmovecircle_jkrobot(2, 1, [-0.442576, 0.00507468, 0.38258, 9.0e+01, 0.0, -9.0e+01, -0.50501907, -0.0458132, 0.38258, 9.0e+01, 0.0, -9.0e+01],
#                                         [-0.50501907, -0.0458132, 0.10520692, 8.97304176e+01, -5.52288735e-03, -8.00018932e+01],
#                                         0.1,
#                                         0.75, 0.75)
    # robot_tcp.end_absmovecircle_jkrobot([-1.13846, 0.0213097, 0.50329, 9.0e+01, 0.0, -9.0e+01],
    #                                     0.1, 0.1)
# 末端速度 cm/s, deg/s
    # robot_tcp.end_vel_jkrobot([0, 0, 0, 0, 0, 5.0])
# 停止并清除任务
    # robot_tcp.stop_jkrobot()
    ############## JKRobot immediate 函数测试 ##############
# 获取末端位姿 m, deg
#     data = robot_tcp.get_end_pos_jkrobot()
#     print(data)
# 获取末端速度 cm/s, deg/s
    # data = robot_tcp.get_end_vel_jkrobot()
    # print(data)
# 获取关节角度 deg
#     data = robot_tcp.get_joint_pos_jkrobot()
#     print(data)
    ############## 夹爪 功能函数测试 ##############
# 吸盘吸气
#     robot_tcp.suck_start(suck_io_id=[2])
# 吸盘破真空
#     robot_tcp.suck_stop(suck_io_id=[2], unsuck_io_id=[3])


















    # 以下功能尚未验证，备份
    ############## enter、exit 函数测试 ##############
    # with RobotTcp('192.168.71.60', 8001) as robot_tcp:
    #     pass
    ############## del 函数测试 ##############
    # robot_tcp = RobotTcp('192.168.71.60', 8001)
    ############## JKRobot 函数测试 ##############
    # robot_tcp.enable_jkrobot() # 使能
    # robot_tcp.disable_jkrobot() # 失能
    # robot_tcp.joints_move_jkrobot([-21.0, -15.0, 116.0, 83.0, 74.0, -88.0]) # 关节角度

    # CNC项目预设点（关节）
    # robot_tcp.joints_move_jkrobot([-10.482145, 23.629826, 130.776079, 26.364802, 79.181900, -89.845420])  # 关节角度
    # robot_tcp.joints_move_jkrobot([-10.464647, -3.093725, 116.709981, 67.138663, 79.326782, -89.999409])  # 关节角度
    # robot_tcp.joints_move_jkrobot([-75.047985, 16.022645, 95.161990, 72.557958, 14.750519, -93.266190])  # 关节角度
    # robot_tcp.joints_move_jkrobot([-75.055877, 22.775084, 102.769670, 58.063679, 14.719620, -93.129387])  # 关节角度
    # CNC项目预设点（末端位姿）
    # robot_tcp.end_absmove_jkrobot([-0.608455, -0.089244, 0.024, 148.436502, 89.195452, -31.511289])
    # robot_tcp.end_absmove_jkrobot([-0.608376, -0.089258, 0.366234, 148.779021, 89.182861, -31.169281])
    # robot_tcp.end_absmove_jkrobot([-0.447302, 0.623228, 0.366234, 148.781578, 89.183280, -31.133523])
    # robot_tcp.end_absmove_jkrobot([-0.447302, 0.623228, 0.216237, 148.781578, 89.183280, -31.133523])




    # robot_tcp.end_absmove_jkrobot([-0.2, -0.65, 0.6, 180, 0, 45]) # 末端位置
    # robot_tcp.end_vel_jkrobot([0, 0, 0, 0, 0, 5.0]) # 末端速度
    # robot_tcp.end_absmove_jkrobot([-0.18, -0.69, 0.622, 180, 0, 90]) # 末端绝对 位移
    # robot_tcp.end_relmove_jkrobot([0.05, 0.1, 0.1, 0, 0, 0]) # 末端相对 位移 相对当前末端坐标系
    # robot_tcp.stop_jkrobot() # 停止并清除任务

    ############## JKRobot immediate 函数测试 ##############
    # data = robot_tcp.get_end_pos_jkrobot()
    # print(data)
    # data = robot_tcp.get_end_vel_jkrobot()
    # print(data)
    # data = robot_tcp.get_joint_pos_jkrobot()
    # print(data)

    ############## AGV 功能函数测试 ##############
    # robot_tcp.agv_nav_location([-0.1, 0.3, -45.5]) # 根据坐标导航
    # robot_tcp.agv_nav_station(1) # 根据站点导航
    # robot_tcp.agv_forward() # 前进100ms
    # time.sleep(0.1)
    # robot_tcp.agv_back() # 后退100ms
    # time.sleep(0.1)
    # robot_tcp.agv_left() # 向左100ms
    # time.sleep(0.1)
    # robot_tcp.agv_right() # 向右100ms;
    ############## 夹爪 功能函数测试 ##############
    # robot_tcp.grasp_close([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # 吸盘吸气
    # robot_tcp.grasp_open([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # 吸盘破真空






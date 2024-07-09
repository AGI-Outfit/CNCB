import numpy as np
import socket
from PIL import Image


class Camera:
    def __init__(self, server_ip, server_port):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((server_ip, server_port))

        self.color_width, self.color_height, self.depth_width, self.depth_height = self.__getParameter()

    def __del__(self):
        self.client_socket.close()

    def __getParameter(self):
        # 发送数据
        message = "parameter\n"
        self.client_socket.send(message.encode())
        # 接收服务器响应
        # 接收数据
        packet = self.client_socket.recv(4096)
        return np.frombuffer(packet, dtype=np.uint32)

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

    def getDepthImage(self) -> np.array:
        message = "depth\n"
        self.client_socket.send(message.encode())
        depth_data = self.receive_all_data(self.depth_width * self.depth_height * 2)
        depth_array = np.frombuffer(depth_data, dtype=np.uint16)
        depth_array.resize((self.depth_height, self.depth_width))
        return depth_array

    def getColorImage(self):
        message = "color\n"
        self.client_socket.send(message.encode())
        color_data = self.receive_all_data(self.color_width * self.color_height * 3)
        color_array = np.frombuffer(color_data, dtype=np.uint8)
        color_array.resize((self.color_height, self.color_width, 3))
        return color_array


def get_image_and_show(camera: Camera):
    depth_array = camera.getDepthImage()
    depth_array = depth_array / depth_array.max() * 255
    depth_image = Image.fromarray(depth_array)
    print("depth", depth_array.shape)

    #color data
    color_array = camera.getColorImage()
    print("color", color_array.shape)

    # to images 1: byte to numpy to image
    color_image = Image.fromarray(color_array)
    # color_image.save(f"./data/color_image_{time_string}.jpg")

    # to images 2: byte to image
    # color_image = Image.frombytes('RGB', (parameter_array[0], parameter_array[1]), color_data)
    color_image.show()
    depth_image.show()


def get_image_and_show_while(camera: Camera):
    import cv2
    color_window = 'color_window'
    depth_window = 'depth_window'
    cv2.namedWindow(color_window, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(depth_window, cv2.WINDOW_AUTOSIZE)
    while True:
        depth_array = camera.getDepthImage()
        depth_array = depth_array / depth_array.max()
        color_array = camera.getColorImage()

        # 在OpenCV窗口中显示图像
        cv2.imshow(color_window, color_array)
        cv2.imshow(depth_window, depth_array)

        # 等待按键或退出事件，1表示等待1ms后继续（非阻塞）
        key = cv2.waitKey(1)

        # 如果按下'q'键，则退出循环
        if key == ord('q'):
            break
        # 销毁所有OpenCV窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    server_ip = '192.168.71.60'
    server_port = 8233
    camera = Camera(server_ip, server_port)
    get_image_and_show_while(camera)
    # get_image_and_show(camera)
    # while True:
    #     get_image_and_show(camera)

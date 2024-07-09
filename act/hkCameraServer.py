import socket
import threading
import numpy as np

import visual.hkCamera as hkCamera


def handle_client(conn, addr, camera):
    print(f'Connected by {addr}')
    while True:
        data = conn.recv(1024).decode()
        if not data:
            break

        if data.strip() == 'color':
            image = camera.getColorImage()
            conn.sendall(image.tobytes())
        elif data.strip() == 'parameter':
            parameter = np.array([640, 480, 0, 0], dtype=np.int32)
            conn.sendall(parameter.tobytes())
    conn.close()
    print(f'{addr} closed')


def main():
    HOST = '0.0.0.0'
    PORT = 8234
    camera = hkCamera.hkCamera()

    # 创建socket并绑定到指定地址和端口
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f'Server is listening on port {PORT}...')

        while True:
            conn, addr = s.accept()
            client_thread = threading.Thread(target=handle_client, args=(conn, addr, camera))
            client_thread.start()


if __name__ == "__main__":
    main()

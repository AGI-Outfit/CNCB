# -- coding: utf-8 --
import threading
import time

import numpy as np
from PIL import Image
from visual.MvImport.MvCameraControl_class import *


class hkCamera():
    def __init__(self):
        self.__stop_camera_flag = False
        self.is_close = False
        self.imageLen = 0
        self.data_lock = threading.Lock()
        self.__init_camera()
        self.start_camera_thread()

    def __init_camera(self):
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE

        # ch:枚举设备 | en:Enum device
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            raise Exception(f"enum devices fail! ret[0x{ret}]")

        if deviceList.nDeviceNum == 0:
            raise Exception(f"find no device!")

        nConnectionNum = 0
        # ch:创建相机实例 | en:Creat Camera Object
        self.cam = MvCamera()

        # ch:选择设备并创建句柄| en:Select device and create handle
        stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            raise Exception(f"create handle fail! ret[0x{ret}]")

        # ch:打开设备 | en:Open device
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise Exception(f"open device fail! ret[0x{ret}]")

        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    raise Exception(f"Set Packet Size fail! ret[0x{ret}]")
            else:
                raise Exception(f"Get Packet Size fail! ret[0x{nPacketSize}]")

        # ch:设置触发模式为off | en:Set trigger mode as off
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            raise Exception(f"set trigger mode fail! ret[0x{ret}]")

        # ch:设置曝光自动曝光off
        ret = self.cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
        if ret != 0:
            raise Exception(f"set exposure auto mode fail! ret[0x{ret}]")
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", 25000)
        if ret != 0:
            raise Exception(f"set exposure time fail! ret[0x{ret}]")

        # ch:获取数据包大小 | en:Get payload size
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            raise Exception(f"get payload size fail! ret[0x{ret}]")
        self.nPayloadSize = stParam.nCurValue

        # ch:开始取流 | en:Start grab image
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise Exception(f"start grabbing fail! ret[0x{ret}]")

        try:
            self.camera_thread = threading.Thread(target=self.__run_camera_thread)
        except:
            raise Exception("error: unable to start thread")

    def start_camera_thread(self):
        self.__stop_camera_flag = False
        self.camera_thread.start()

    def __run_camera_thread(self):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        while not self.__stop_camera_flag:
            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if ret == 0:
                nRGBSize = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 3
                self.imageLen = nRGBSize

                stConvertParam = MV_CC_PIXEL_CONVERT_PARAM_EX()
                memset(byref(stConvertParam), 0, sizeof(stConvertParam))
                stConvertParam.nWidth = stOutFrame.stFrameInfo.nWidth
                stConvertParam.nHeight = stOutFrame.stFrameInfo.nHeight
                stConvertParam.pSrcData = stOutFrame.pBufAddr
                stConvertParam.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen
                stConvertParam.enSrcPixelType = stOutFrame.stFrameInfo.enPixelType
                stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
                stConvertParam.pDstBuffer = (c_ubyte * nRGBSize)()
                stConvertParam.nDstBufferSize = nRGBSize

                ret = self.cam.MV_CC_ConvertPixelTypeEx(stConvertParam)
                if ret != 0:
                    print("convert pixel fail! ret[0x%x]" % ret)
                    sys.exit()

                self.cam.MV_CC_FreeImageBuffer(stOutFrame)

                img_buff = (c_ubyte * stConvertParam.nDstLen)()
                memmove(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                data = np.frombuffer(img_buff, count=int(stConvertParam.nDstLen), dtype=np.uint8)
                data = data.reshape(stConvertParam.nHeight, stConvertParam.nWidth, 3)
                self.data_lock.acquire()
                self.image = np.array(data)
                self.data_lock.release()
            else:
                print("get one frame fail, ret[0x%x]" % ret)

    def getColorImage(self):
        self.data_lock.acquire()
        image_array = np.array(self.image)
        self.data_lock.release()
        image_pil = Image.fromarray(image_array)
        image_pil = image_pil.resize((640, 480))
        return np.array(image_pil)

    def getColorImageFullSize(self):
        self.data_lock.acquire()
        image_array = np.array(self.image)
        self.data_lock.release()
        return image_array

    def __del__(self):
        if self.is_close is True:
            return
        self.__stop_camera_flag = True
        self.camera_thread.join()

        # ch:停止取流 | en:Stop grab image
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            raise Exception(f"stop grabbing fail! ret[0x{ret}]")

        # ch:关闭设备 | Close device
        ret = self.cam.MV_CC_CloseDevice()
        if ret != 0:
            raise Exception(f"close device fail! ret[0x{ret}]")

        # ch:销毁句柄 | Destroy handle
        ret = self.cam.MV_CC_DestroyHandle()
        if ret != 0:
            print("destroy handle fail! ret[0x%x]" % ret)
            raise Exception(f"destroy handle fail! ret[0x{ret}]")
        self.is_close = True


if __name__ == "__main__":
    camera = hkCamera()

    time.sleep(3)
    import cv2

    cv2.imwrite('test.png', camera.getColorImage())

    camera.__del__()

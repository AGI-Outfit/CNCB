from .base_tool import BaseTool

import pyaudio
import wave
import whisper
import numpy as np
import io
import base64
import scipy.io.wavfile as wavfile
import whisper
import os
import datetime
from typing import Union

class SpeechRecognizer(BaseTool):
    def __init__(self, args):
        super().__init__(args)
        self.args_speech_recog = self.args["speechrecog"]
        self.recoder = pyaudio.PyAudio()
        self.model = whisper.load_model(self.args_speech_recog["model_name"])

        # 定义录音相关的参数
        self.CHUNK = 1024  # 每次读取的音频数据块大小
        self.FORMAT = pyaudio.paInt16  # 音频数据格式
        self.CHANNELS = 2  # 音频通道数
        self.RATE = 44100  # 录音采样率
        self.RECORD_SECONDS = self.args_speech_recog["record_time"]  # 录音时长
        self.use_speechrecog = self.args_speech_recog["use_speechrecog"]

        self.level_print("Success: Speech Recognizer Init.", 1)

    def record(self) -> Union[str, None]:
        if self.use_speechrecog:
            stream = self.recoder.open(format=self.FORMAT,
                                                    channels=self.CHANNELS,
                                                    rate=self.RATE,
                                                    input=True,
                                                    frames_per_buffer=self.CHUNK)
            self.level_print("开始监听指令...", 0)
            frames = []
            for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                data = stream.read(self.CHUNK)
                frames.append(data)
            self.level_print("结束。", 0)

            # 关闭音频流
            stream.stop_stream()
            stream.close()

            # 打开一个wave文件，用于写入录制的音频数据
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_path = self.args_speech_recog["wave_output_path"]
            save_path = file_path + "record-" + current_time + ".wav"
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            wf = wave.open(save_path, 'wb')
            # 设置wave文件的参数
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.recoder.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            # 将录制的音频数据写入wave文件
            wf.writeframes(b''.join(frames))
            # 关闭wave文件
            wf.close()
            return save_path
        return None

    def speech_recog(self, audio_path: str) -> str:
        if self.use_speechrecog:
            self.level_print("开始识别指令...", 0)
            result = self.model.transcribe(audio_path)
            self.level_print("识别结果： {0}".format(result["text"]), 0)
            return result["text"]
        return ""


    def __del__(self):
        self.recoder.terminate()
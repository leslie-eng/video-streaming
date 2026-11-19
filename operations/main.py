# you can either use RTSP IP or HTTP

import cv2
import requests
import urllib3
import logging


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

stream_url = "https://192.168.100.7:8080/video"


class VideoStream:
    def __init__(self, stream_url, timeout=20.0) -> None:
        self.stream_url = stream_url
        self.timeout = timeout
    
    def HTTP_Conn(self):
        pass
    
    def RTSP_Conn(self, login, passwd):
        IP_ADDRESS = self.stream_url.split("/")[2]
        conn_url = f"rtsp://{login}:{passwd}@{IP_ADDRESS}"
    
    
if __name__ == "__main__":
    stream = VideoStream(stream_url=stream_url)
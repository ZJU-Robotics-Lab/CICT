# -*- coding: utf-8 -*-
import socket
from flask import Flask, render_template, Response

address = '0.0.0.0'
port = 8080

class Camera(object):
    def __init__(self):
        self.svrsocket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
       # use socket to transmit data
        self.svrsocket.bind((address, port))
       # boud IP
      
    def get_frame(self):
       # 定义获取帧的方法，使用socket接收数据
        data, address=self.svrsocket.recvfrom(65535)
        print('get data:', len(data))
        return data

app = Flask(__name__)
 
@app.route('/')
def index():
    """return main page"""
    return render_template('index.html')
 
def gen(camera):
    """video"""
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
               b'Content-Type:image/jpeg\r\n\r\n'+ frame +b'\r\n')
 
@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace;boundary=frame')
 
if __name__ =='__main__':
    app.run(host='0.0.0.0', port = port, debug = True)
    # default port: 5000
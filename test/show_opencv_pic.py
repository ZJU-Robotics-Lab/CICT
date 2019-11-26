import cv2
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
import random

HEIGHT = 1080
WIDTH = 1920
app = QtGui.QApplication([])

## Create window with GraphicsView widget
win = pg.GraphicsLayoutWidget()
win.show()  ## show widget alone in its own window
win.setWindowTitle('Show OpenCV Vedio')
view = win.addViewBox()

## lock the aspect ratio so pixels are always square
view.setAspectLocked(True)

## Create image item
img = pg.ImageItem(border='w')
view.addItem(img)

## Set initial view bounds
view.setRange(QtCore.QRectF(0, 0, HEIGHT, WIDTH))

i = 0
cap = cv2.VideoCapture(0)
updateTime = ptime.time()
fps = 0

def updateData():
    global img, updateTime, fps
    ret, data = cap.read()
    data = cv2.resize(data, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    ## Display the data
    img.setImage(data)
    # i = (i+1) % data.shape[0]

    QtCore.QTimer.singleShot(1, updateData)
    now = ptime.time()
    fps2 = 1.0 / (now-updateTime)
    updateTime = now
    fps = fps * 0.9 + fps2 * 0.1
    
    #print "%0.1f fps" % fps
    

updateData()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

print("importing modules")

import cv2
import numpy as np

import argparse
from time import perf_counter

tStarts = []
def startTimer():
    if args.debug_timing:
        tStarts.append(perf_counter())
def endTimer(name):
    if args.debug_timing:
        print(f"{'    '*len(tStarts)}{name} took {int((perf_counter()-tStarts.pop())*10000)/10} ms")



#########################################
############### processing ##############
#########################################

def preparePoints(pts):
    pts = np.asarray(pts).reshape(-1, 3)
    return np.concatenate((pts, np.ones(pts.shape[:-1]+(1,), dtype=pts.dtype)), axis=-1)

IMG_WIDTH, IMG_HEIGHT = 400, 400
BG_GRAY = 80
AXES_POINTS = preparePoints(np.stack(np.mgrid[0:257:256, 0:257:256, 0:257:256], axis=-1))
yaw, pitch = np.pi*7/6, 0.5
lastFrameTime = perf_counter()
frameAverage = None
def process(frame):
    assert frame.dtype == np.uint8
    frame = cv2.resize(frame, (0,0), fx=1/args.scale, fy=1/args.scale)
    
    # average
    global frameAverage
    if frameAverage is None:
        frameAverage = frame.astype(np.float32)
    else:
        cv2.accumulateWeighted(frame, frameAverage, 0.1)
        frame = frameAverage.astype(np.uint8)
    cv2.imshow("frame", frame)
    
    startTimer()
    cs = COLOR_SPACES[colorSpace]
    
    # rotate
    global yaw, pitch, lastFrameTime
    now = perf_counter()
    dt, lastFrameTime = now - lastFrameTime, now
    if rotate:
        yaw += 0.2 * dt
    
    # construct the transformation matrix
    cosY, sinY = np.cos(yaw), np.sin(yaw)
    cosP, sinP = np.cos(pitch), np.sin(pitch)
    transform_matrix = np.array([
        [IMG_WIDTH*3.3, 0,              0],
        [0,             IMG_HEIGHT*3.3, 0],
        [0,             0,              1]
    ], dtype=np.float32) @ np.array([
        [1, 0,     0,    0  ],
        [0, cosP, -sinP, 0  ],
        [0, sinP,  cosP, 1.5]
    ], dtype=np.float32) @ np.array([
        [cosY, 0, -sinY, 0],
        [0,   -1,  0,    0],
        [sinY, 0,  cosY, 0],
        [0,    0,  0,    1]
    ], dtype=np.float32) @ np.array([
        [1, 0, 0, -128 ],
        [0, 1, 0, -128 ],
        [0, 0, 1, -128 ],
        [0, 0, 0,  1000]
    ], dtype=np.float32)
    # print(transform_matrix.astype(np.int))
    data_transform = np.array([
        [cs[3], 0, 0, 0],
        [0,     1, 0, 0],
        [0,     0, 1, 0],
        [0,     0, 0, 1]
    ], dtype=np.float32)
    
    def projectPoints(pts, isData=False):
        startTimer()
        try:
            startTimer()
            pts = pts.dot((transform_matrix@data_transform if isData else transform_matrix).T)
            endTimer("transform")
            
            startTimer()
            pointsXY = np.uint32(pts[...,:2]/pts[...,2,np.newaxis] + [IMG_WIDTH/2, IMG_HEIGHT/2]).T
            endTimer("divide & shift")
            if isData:
                startTimer()
                sort = np.argsort(pts[:,2], kind="quicksort")[::-1]
                pointsXY = pointsXY[:,sort]
                endTimer("argsort")
                return pointsXY, sort
            else:
                return pointsXY
        finally:
            endTimer(f"projectPoints ({len(pts)})")
    
    # get the point cloud
    global colors, colorCoords, csDirty
    if not frozen or csDirty:
        startTimer()
        colors = frame.reshape(-1, 3)
        
        startTimer()
        
        print(len(colors), "colors")
        startTimer()
        colorIndices = np.int32(np.right_shift(colors, 1))
        colorIndices = colorIndices[:,0] | np.left_shift(colorIndices[:,1], 8) | np.left_shift(colorIndices[:,2], 16)
        endTimer("make colorIndices")
        startTimer()
        _, colorIndices = np.lib.arraysetops.unique(colorIndices, return_index=True)
        endTimer("np.lib.arraysetops.unique")
        colors = colors[colorIndices]
        print(len(colors), "unique colors")
        
        endTimer("color binning")
        
        # colors = np.stack(np.mgrid[0:256:13, 0:256:13, 0:256:13], axis=-1).reshape(-1, 3).astype(np.uint8)
        colorCoords = cv2.cvtColor(colors[:,np.newaxis], cs[0]).reshape(-1, 3)
        colorCoords = colorCoords[:,cs[2]]
        colorCoords = preparePoints(colorCoords)
        csDirty = False
        endTimer("get points cloud")
    
    img = np.full((IMG_HEIGHT, IMG_WIDTH, 3), BG_GRAY, dtype=np.uint8)
    
    # draw the RGB axes
    pointsXY = projectPoints(AXES_POINTS)
    cv2.line(img, (pointsXY[0,0],pointsXY[1,0]), (pointsXY[0,1],pointsXY[1,1]), (255,0,0))
    cv2.line(img, (pointsXY[0,0],pointsXY[1,0]), (pointsXY[0,2],pointsXY[1,2]), (0,255,0))
    cv2.line(img, (pointsXY[0,0],pointsXY[1,0]), (pointsXY[0,4],pointsXY[1,4]), (0,0,255))
    img[pointsXY[1], pointsXY[0]] = np.full(pointsXY.shape[1:]+(3,), 255)
    
    # draw the point cloud
    pointsXY, sort = projectPoints(colorCoords, isData=True)
    img[pointsXY[1], pointsXY[0]] = colors[sort]
    
    if IMG_WIDTH <= 800:
        scale = 800//IMG_WIDTH
        inter = cv2.INTER_NEAREST
    else:
        scale = 800/IMG_WIDTH
        inter = cv2.INTER_CUBIC
    cv2.imshow("img", cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=inter))
    
    endTimer("process")



##########################################
######### command-line arguments #########
##########################################

parser = argparse.ArgumentParser(description="Live plotting of color distribution")
optGroup = parser.add_mutually_exclusive_group()
optGroup.add_argument("-d", "--device", type=int, default=0, metavar="ID",
                      help="device ID of the camera to use (default: %(default)s)")
optGroup.add_argument("-i", "--input-image", metavar="FILE", help="optional image to use instead of a live camera")
optGroup.add_argument("-v", "--input-video", metavar="FILE", help="optional video to use instead of a live camera")
parser.add_argument("-s", "--scale", type=float, default=3.0, metavar="FACTOR",
                    help="amount to downsample each frame (default: %(default)s)")
debugGroup = parser.add_argument_group(title="debug flags")
debugGroup.add_argument("--debug-timing", action="store_true", help="print how long various operations take")
args = parser.parse_args()



#########################################
############### main code ###############
#########################################

colorSpace, csDirty = 0, True
def makeCS(name, code, scale0=1, order=(0,1,2)):
    return (code, name, order, scale0)
COLOR_SPACES = [
    makeCS("RGB",cv2.COLOR_BGR2RGB),
    makeCS("HSV",cv2.COLOR_BGR2HSV,scale0=255/180,order=(0,2,1)),
    makeCS("HLS",cv2.COLOR_BGR2HLS,scale0=255/180),
    makeCS("CIE L*a*b*",cv2.COLOR_BGR2Lab,order=(1,0,2)),
    makeCS("CIE L*u*v*",cv2.COLOR_BGR2Luv,order=(1,0,2)),
    makeCS("YCrCb",cv2.COLOR_BGR2YCrCb,order=(1,0,2)),
    makeCS("CIE XYZ",cv2.COLOR_BGR2XYZ,order=(1,0,2))
]

frozen = False
rotate = False

lastMX, lastMY = 0, 0
def onMouse(event, x, y, flags, param):
    global lastMX, lastMY, pitch, yaw
    if flags & cv2.EVENT_FLAG_LBUTTON:
        dx, dy = x - lastMX, y - lastMY
        yaw += dx*0.01
        pitch += dy*0.01
    lastMX, lastMY = x, y

def onKey(key):
    print(f"key: {key}")
    if key == ord("d") or key == ord("D"):
        args.device += 1
        print(f"    switching device id to {args.device}")
        global cap
        cap = initCapture()
    if key == ord("s") or key == ord("S"):
        cap.set(cv2.CAP_PROP_SETTINGS, 1)
    if key == ord("f") or key == ord("F"):
        global frozen
        frozen = not frozen
    if key == ord("r") or key == ord("R"):
        global rotate
        rotate = not rotate
    if key == ord("b") or key == ord("B"):
        global BG_GRAY
        BG_GRAY += 256/10
        BG_GRAY %= 256
    
    def changeColorSpace(d):
        global colorSpace, csDirty
        colorSpace += d
        colorSpace %= len(COLOR_SPACES)
        csDirty = True
        print("Color space:", COLOR_SPACES[colorSpace][1])
    if key == ord("j") or key == ord("J"):
        changeColorSpace(-1)
    if key == ord("k") or key == ord("K"):
        changeColorSpace(+1)
    
    SCALE_FACTOR = 1.1
    def scaleImg(f):
        global IMG_WIDTH, IMG_HEIGHT
        IMG_WIDTH, IMG_HEIGHT = (int(d*f) for d in (IMG_WIDTH, IMG_HEIGHT))
    if key == ord("-"):
        scaleImg(1/SCALE_FACTOR)
    if key == ord("="):
        scaleImg(SCALE_FACTOR)

######### VideoCapture #########
def initCapture():
    if args.input_image is not None: return None
    print("Initializing VideoCapture...")
    if args.input_video is not None:
        cap = cv2.VideoCapture(args.input_video)
    else:
        cap = cv2.VideoCapture(args.device)
        if not cap.isOpened():
            print(f"    failed to open camera device {args.device}")
            print(f"    resetting device id to 0")
            args.device = 0
            return initCapture()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print("    done.")
    return cap
cap = initCapture()

if args.input_image is not None:
    inputImage = cv2.imread(args.input_image)

try:
    frameCount, lastSecond = 0, perf_counter()
    while True:
        # read the next frame and make sure it's valid
        if args.input_image is not None:
            frame = inputImage.copy()
        elif not frozen:
            ret, frame = cap.read()
            def isFrameOK():
                if not ret or frame is None:
                    return False
                for i in [0,1,2]:
                    if cv2.countNonZero(frame[:,:,i]) > 0:
                        return True
                return False
            if not isFrameOK():
                print("Got a bad frame, reinitializing.")
                cap.release()
                cap = initCapture() # reopen the VideoCapture
                continue
        
        process(frame)
        cv2.setMouseCallback("img", onMouse)
        
        frameCount += 1
        if perf_counter() - lastSecond > 1.0:
            lastSecond += 1.0
            cv2.setWindowTitle("frame", f"{frameCount} FPS")
            frameCount = 0
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key != 0xFF:
            onKey(key)
finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
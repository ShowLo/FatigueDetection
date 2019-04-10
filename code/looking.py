# -*- coding: UTF-8 -*-

import cv2
import imageio
import dlib
import os
import numpy as np
import math
import imutils
from imutils import face_utils

def getRotationVector(sixPoints, imageSize):
    '''
    计算旋转向量
    sixPoints : 六个特征点
    imageSize : 图像尺寸
    '''
    # 3D模型点
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)])
     
    # 相机的内参数
    # 把焦距近似为图像的宽度
    focalLength = imageSize[1]
    # 用图像中心逼近光学中心
    center = (imageSize[1]/2, imageSize[0]/2)
    # 相机矩阵
    cameraMatrix = np.array([[focalLength, 0, center[0]], [0, focalLength, center[1]], [0, 0, 1]], dtype = 'double')

    # 失真系数，假设无失真
    distCoeffs = np.zeros((4,1))
    # 求解旋转向量
    (success, rotationVector, translationVector) = cv2.solvePnP(model_points, sixPoints, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    return success, rotationVector

def rotationToYaw(rotationVector):
    '''
    把旋转向量转为欧拉角，返回偏航角
    rotationVector : 旋转向量
    '''
    # 旋转角
    theta = cv2.norm(rotationVector, cv2.NORM_L2)
    
    # 四元数
    w = math.cos(theta / 2)
    x = math.sin(theta / 2) * rotationVector[0][0] / theta
    y = math.sin(theta / 2) * rotationVector[1][0] / theta
    z = math.sin(theta / 2) * rotationVector[2][0] / theta
    
    # 偏航角
    temp = 2.0 * (w * y - z * x)
    if temp > 1.0:
        temp = 1.0
    if temp < -1.0:
        temp = -1.0
    yaw = math.asin(temp)
    
	# 将弧度转换为度
    return yaw / math.pi * 180

def calculateYaw(sixPoints, imageSize):
    '''
    计算偏航角
    sixPoints : 六个特征点
    imageSize : 图像尺寸
    '''
    success, rotationVector = getRotationVector(sixPoints, imageSize)
    if success:
        return success, rotationToYaw(rotationVector)
    else:
        print('Fail to get the rotationVector!')
        return success, None

def getYaw(image, rotateTimes = 0, rotateAngle = 5, rotateDirection = 'clockwise'):
    '''
    计算人脸的偏航角
    image : 输入的灰度图像
    rotateTimes : 旋转次数
    rotateAngle : 旋转角度
    rotateDirection : 旋转方向
    '''
    img = None
    if rotateTimes == 0:
        img = image
    elif rotateDirection == 'clockwise':
        print('Try rotate %s for %d degree' % (rotateDirection, rotateAngle * rotateTimes))
        img = imutils.rotate_bound(image, rotateAngle * rotateTimes)
    else:
        print('Try rotate %s for %d degree' % (rotateDirection, rotateAngle * rotateTimes))
        img = imutils.rotate_bound(image, -rotateAngle * rotateTimes)
    # 鼻尖
    noseTip = 30
    # 下巴
    chin = 8
    # 右眼的右眼角（以照片中人的左右为基准）
    rightEyeRightCorner = 36
    # 左眼的左眼角
    leftEyeLeftCorner = 45
    # 右嘴角
    rightMouthCorner = 48
    # 左嘴角
    leftMouthCorner = 54

    # 人脸提取器
    detector = dlib.get_frontal_face_detector()
    # 人脸68个特征点的检测器
    predictor = dlib.shape_predictor('model\\shape_predictor_68_face_landmarks.dat')
    # 检测人脸，第二个参数中的1表示应该向上采样图像1次，这会使所有的东西都变大，能发现更多的面孔
    faces = detector(img, 1)
    # 检测到了人脸
    if len(faces) > 0:
        print('Faces detected.')
        face = faces[0]
        
        # 寻找人脸的68个标定点
        shape = predictor(img, face)
        points = face_utils.shape_to_np(shape)
        
        # 取出需要的六个特征点
        sixPoints = np.array([points[noseTip], points[chin], points[rightEyeRightCorner], points[leftEyeLeftCorner], points[rightMouthCorner], points[leftMouthCorner]], dtype='double')
        # 计算偏航角
        success, yaw = calculateYaw(sixPoints, img.shape)
        
        '''
        # 遍历特征点，圈出来
        for pt in sixPoints:
            pt_pos = (int(pt[0]), int(pt[1]))
            cv2.circle(img, pt_pos, 1, (255, 0, 0), 1)
        # 在图片中标注人脸，并显示
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        # 框出人脸
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 3)
        cv2.imshow("image", img)
        cv2.waitKey(100)
        '''

        # 返回偏航角
        return success, yaw
    else:
        rotateTimes += 1
        if rotateDirection == 'clockwise':
            if rotateTimes > 5:
                # 顺时针旋转多次仍未检出人脸，改用逆时针旋转
                return getYaw(image, rotateTimes=1, rotateDirection='anticlockwise')
            else:
                # 顺时针旋转再检测
                return getYaw(image, rotateTimes=rotateTimes, rotateDirection='clockwise')
        else:
            if rotateTimes > 5:
                print('No faces detected though rotate for several times, break!')
                return False, None
            else:
                # 逆时针旋转再次检测
                return getYaw(image, rotateTimes=rotateTimes, rotateDirection='anticlockwise')


def looking_detection(videoPath, output_dir, normalMinYaw, normalMaxYaw, windowLen):
    '''
    检测是否四处张望
    videoPath : 视频地址
    output_dir : 输出视频地址
    normalMinYaw : 正常的偏航角的最小值
    normalMaxYaw : 正常的偏航角的最大值
    windowLen : 滑动窗口长度
    '''
    cap = cv2.VideoCapture(videoPath)

    # 视频编解码器，MPEG-4.2编码类型
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    fps =int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(output_dir, fourcc, fps, size)

    # 记录是否四处张望
    notLookingAround = []
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # 转成灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            (height, width) = gray.shape
            # 计算偏航角
            success, yaw = getYaw(gray)
            if success:
                # 在正常的角度范围内
                if yaw > normalMinYaw and yaw < normalMaxYaw:
                    notLookingAround.append(1)
                    # 总帧数还不到滑动窗口长或者滑动窗口内出现足够多次正常状态的话可以认为处于正常状态
                    if count < windowLen or np.sum(np.array(notLookingAround[count - windowLen : ])) > windowLen // 2:
                        cv2.putText(frame, 'Normal', (round(0.1 * width), round(0.1 * height)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.FONT_HERSHEY_SIMPLEX)
                    # 滑动窗口内的正常状态不够多，仍然认为处于四处观望状态
                    else:
                        cv2.putText(frame, 'Looking around', (round(0.1 * width), round(0.1 * height)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.FONT_HERSHEY_SIMPLEX)
                else:
                    notLookingAround.append(0)
                    # 超过正常角度范围，认为处于四处张望状态
                    cv2.putText(frame, 'Looking around', (round(0.1 * width), round(0.1 * height)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.FONT_HERSHEY_SIMPLEX)
            else:
                print('Fail to get yaw!')
                notLookingAround.append(0)
                # 如果没检测到人脸，大概率处于四处张望状态
                cv2.putText(frame, 'Looking around', (round(0.1 * width), round(0.1 * height)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.FONT_HERSHEY_SIMPLEX)

            '''
            cv2.imshow('f', frame)
            cv2.waitKey(100)
            '''

            # 写入新的视频文件
            videoWriter.write(frame)
        else:
            break
        count += 1
    videoWriter.release()
    
    '''
    import matplotlib.pyplot as plt
    print(yawList)
    plt.plot(range(len(yawList)), yawList, 'g-s')
    plt.show()
    '''

def main():
    videoPath = 'Fatigue Detection/looking'
    output_dir = 'output/looking'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = os.listdir(videoPath)
    normalMinYaw = -35
    normalMaxYaw = -25
    windowLen = 3
    
    for f in files:
        looking_detection(os.path.join(videoPath, f), os.path.join(output_dir, f), normalMinYaw, normalMaxYaw, windowLen)
    
    '''
    f = files[2]
    looking_detection(os.path.join(videoPath, f), os.path.join(output_dir, f), normalMinYaw, normalMaxYaw)
    '''

if __name__ == "__main__":
    main()
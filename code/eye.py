# -*- coding: UTF-8 -*-

import cv2
import skimage
import imageio
import dlib
import os
import numpy as np
import imutils
from imutils import face_utils
from scipy.spatial import distance

def EAR(eye):
    '''
    计算EAR值
    eye : 眼睛的六个特征点
    '''
    P26 = distance.euclidean(eye[1], eye[5])
    P35 = distance.euclidean(eye[2], eye[4])
    P14 = distance.euclidean(eye[0], eye[3])
    return (P26 + P35) / (2.0 * P14)

def regionPart(region, eye):
    '''
    判断dlib检测到的眼睛位于haar检测到的眼睛区域的哪一部分
    region : haar检测到的眼睛区域
    eye : dlib检测到的眼睛
    '''
    count = 0
    ySum = 0
    # 如果dlib检测到的6个特征点有超过3个位于haar眼睛区域内，认为位于其内，否则认为在其外
    for i in range(6):
        ySum += eye[i][1]
        # 判断特征点是否位于haar眼睛区域内
        if eye[i][0] >= region[0] and eye[i][0] <= region[2] and eye[i][1] >= region[1] and eye[i][1] < region[3]:
            count += 1
    yMean = ySum / 6.0
    yRange = region[3] - region[1]
    if count <= 3:
        return 'out'
    # 以垂直方向的位置判断dlib检测到的眼睛是位于haar眼睛区域的中间还是偏外边
    elif (yMean - region[1] >= yRange / 3.0) and (yMean - region[1] <= yRange * 2 / 3.0):
        return 'middle'
    else:
        return 'side'

def penalty_based_part(leftEye, rightEye, eyes, left, top):
    '''
    根据dlib检测到的眼睛位于haar检测器检测到的眼睛区域的哪一部分返回相应的惩罚系数
    leftEye : dlib检测到的左眼
    rightEye : dlib检测到的右眼
    eyes : haar检测到的眼睛区域
    left, top : 人脸区域左上角坐标
    '''
    
    # 存放haar检测器检测到的眼睛区域
    eyesRegion = []
    for (ex, ey, ew, eh) in eyes:
        eyesRegion.append([left + ex, top + ey, left + ex + ew, top + ey + eh])
    # 按照检测到的眼睛区域面积大小排序
    eyesRegion.sort(key = lambda x : (x[2] - x[0]) * (x[3] - x[1]))
    # 根据dlib检测到的眼睛位于haar检测器检测到的眼睛区域的哪一部分返回相应的惩罚系数
    temp = []
    for region in eyesRegion:
        partLeft = regionPart(region, leftEye)
        partRight = regionPart(region, rightEye)
        # 位于中间，放回最小的1，相当于不惩罚
        if partLeft == 'middle' or partRight == 'middle':
            return 1
        else:
            temp.append(partLeft)
            temp.append(partRight)
    # 位于偏外边
    if 'side' in temp:
        return 2
    # 不在haar检测到的眼睛区域内
    else:
        return 4

def getEAR(image, hasGlasses, rotateTimes = 0, rotateAngle = 5, rotateDirection = 'clockwise'):
    '''
    计算EAR值
    image : 输入的灰度图像
    hasGlasses : 是否戴眼镜
    rotateTimes : 旋转次数
    rotateAngle : 旋转角度
    rotateDiretion : 旋转方向
    origianlImgCopy : 原始图像的备份
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

    # 左右眼在68个点中的索引
    (leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

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
        
        # 取出眼睛对应的特征点
        leftEye = points[leStart : leEnd]
        rightEye = points[reStart : reEnd]
        # 计算眼睛的EAR值
        leftEAR = EAR(leftEye)
        rightEAR = EAR(rightEye)

        '''
        # 遍历所有点，打印出其坐标，并圈出来
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 1, (255, 0, 0), 1)
        cv2.imshow("image", img)
        '''

        # 在图片中标注人脸，并显示
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        # 框出人脸
        # cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 3)
        faceRegion = img[top : bottom, left : right]
        # 在人脸上检测眼睛是否存在
        eyes = eyes_exist_detection(faceRegion, hasGlasses)
        # 对haar检测器检测到的眼睛与dlib检测到的眼睛进行对比，差距过大的进行惩罚
        penalty = 1

        if len(eyes) == 0:
            # 没有检测到眼睛，惩罚系数最大
            penalty = 5
        else:
            penalty = penalty_based_part(leftEye, rightEye, eyes, left, top)
            
        '''
        for (ex, ey, ew, eh) in eyes:
            # 框出眼睛
            cv2.rectangle(faceRegion, (ex, ey), (ex + ew,ey + eh),(0, 255, 0), 2)
            
        # 寻找眼睛轮廓并绘制出来
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img, [leftEyeHull], -1, (255, 0, 0), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (255, 0, 0), 1)
        cv2.imshow('i', img)
        cv2.waitKey(100)
        '''

        # 返回一个修正的EAR值
        return (leftEAR + rightEAR) / (2 * penalty)
    else:
        rotateTimes += 1
        if rotateDirection == 'clockwise':
            if rotateTimes > 5:
                # 顺时针旋转多次仍未检出人脸，改用逆时针旋转
                return getEAR(image, hasGlasses, rotateTimes=1, rotateDirection='anticlockwise')
            else:
                # 顺时针旋转再次检测
                return getEAR(image, hasGlasses, rotateTimes=rotateTimes, rotateDirection='clockwise')
        else:
            if rotateTimes > 5:
                print('No faces detected though rotate for several times, break!')
                return 0
            else:
                # 逆时针旋转再次检测
                return getEAR(image, hasGlasses, rotateTimes=rotateTimes, rotateDirection='anticlockwise')

def eye_close_detection(videoPath, output_dir, threshold, windowLen, hasGlasses = True):
    '''
    检测眼睛是否闭合
    videoPath : 视频地址
    output_dir : 输出视频地址
    threshold : 根据EAR值判断眼睛是否闭合的一个阈值
    windowLen : 滑动窗口长度
    hasGlasses : 是否戴眼镜
    '''
    cap = cv2.VideoCapture(videoPath)

    # 视频编解码器，MPEG-4.2编码类型
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    fps =int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(output_dir, fourcc, fps, size)

    # 记录EAR值
    earList = []
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # 转成灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ear = getEAR(gray, hasGlasses)
            earList.append(ear)
            (height, width) = gray.shape
            
            # 开始的几帧直接按照跟阈值相比的方式判断是闭眼还是睁眼
            if count < windowLen:
                if ear > threshold:
                    cv2.putText(frame, 'Eyes Open', (round(0.1 * width), round(0.1 * height)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.FONT_HERSHEY_SIMPLEX)
                else:
                    cv2.putText(frame, 'Eyes Closed', (round(0.1 * width), round(0.1 * height)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.FONT_HERSHEY_SIMPLEX)
            else:
                # 检测到的眼睛睁开的情况下需要判断在滑动窗口内是否睁眼次数占多，是的话才认为是睁眼，否则仍然认为是闭眼
                if ear > threshold and np.sum(np.array(earList[count - windowLen : ]) > threshold) > windowLen // 2:
                    cv2.putText(frame, 'Eyes Open', (round(0.1 * width), round(0.1 * height)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.FONT_HERSHEY_SIMPLEX)
                else:
                    cv2.putText(frame, 'Eyes Closed', (round(0.1 * width), round(0.1 * height)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.FONT_HERSHEY_SIMPLEX)
            count += 1
            
            '''
            cv2.imshow('f', frame)
            cv2.waitKey(100)
            '''

            # 写入新的视频文件
            videoWriter.write(frame)
            
        else:
            break
    videoWriter.release()
    '''
    import matplotlib.pyplot as plt
    print(earList)
    plt.plot(range(len(earList)), earList, 'g-s')
    plt.show()
    '''

def eyes_exist_detection(faceRegion, hasGlasses):
    '''
    利用opencv自带的haar检测器检测眼睛是否存在
    faceRegion : 人脸
    hasGlasses : 是否戴眼镜
    '''
    xmlPath = 'haar/eye'
    eye_cascade = None
    # 利用opencv提供的眼睛检测器检测眼睛区域
    if hasGlasses:
        # 使用适用于戴眼镜的检测器
        eye_cascade = cv2.CascadeClassifier(os.path.join(xmlPath, 'haarcascade_eye_tree_eyeglasses.xml'))
    else:
        # 使用适用于不戴眼镜的检测器
        eye_cascade = cv2.CascadeClassifier(os.path.join(xmlPath, 'haarcascade_eye.xml'))
    # 检测眼睛
    eyes = eye_cascade.detectMultiScale(faceRegion, 1.1, 2)
    return eyes

def main():
    videoPath = 'Fatigue Detection/eye'
    output_dir = 'output/eye'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = os.listdir(videoPath)
    # 各个人的阈值
    threshold1 = 0.325
    threshold2 = 0.05
    threshold3 = 0.23
    threshold4 = 0.3
    # 各个人的位置索引及是否戴眼镜
    personIndex1 = [0, 2, 5, 9, 10, 17, 18, 19, 26, 28, 31, 32]
    hasGlasses1 = [False if i <= 10 else True for i in personIndex1]
    personIndex2 = [1, 3]
    hasGlasses2 = [True for i in personIndex2]
    personIndex3 = [4, 7, 13, 14, 24, 25, 27]
    hasGlasses3 = [False if i <= 7 else True for i in personIndex3]
    personIndex4 = [6, 8, 11, 12, 15, 16, 20, 21, 22, 23, 29, 30]
    hasGlasses4 = [False if i <= 13 else True for i in personIndex4]

    #滑动窗口长度
    windowLen = 3
    
    # 检测是否闭眼并保存视频
    for i in range(len(personIndex1)):
        eye_close_detection(os.path.join(videoPath, files[personIndex1[i]]), os.path.join(output_dir, files[personIndex1[i]]), threshold1, windowLen, hasGlasses1[i])
    for i in range(len(personIndex2)):
        eye_close_detection(os.path.join(videoPath, files[personIndex2[i]]), os.path.join(output_dir, files[personIndex2[i]]), threshold2, windowLen, hasGlasses2[i])
    for i in range(len(personIndex3)):
        eye_close_detection(os.path.join(videoPath, files[personIndex3[i]]), os.path.join(output_dir, files[personIndex3[i]]), threshold3, windowLen, hasGlasses3[i])
    for i in range(len(personIndex4)):
        eye_close_detection(os.path.join(videoPath, files[personIndex4[i]]), os.path.join(output_dir, files[personIndex4[i]]), threshold4, windowLen, hasGlasses4[i])
        
    #eye_close_detection(os.path.join(videoPath, files[4]), os.path.join(output_dir, files[4]), threshold3, windowLen, True)

if __name__ == "__main__":
    main()
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

def MAR(mouth):
    '''
    计算MAR值(mouth aspect ratio)，嘴巴纵横比
    mouth : 嘴巴的特征点
    '''
    P13_P19 = distance.euclidean(mouth[13], mouth[19])
    P14_P18 = distance.euclidean(mouth[14], mouth[18])
    P15_P17 = distance.euclidean(mouth[15], mouth[17])
    P12_P16 = distance.euclidean(mouth[12], mouth[16])
    return (P13_P19 + P14_P18 + P15_P17) / (3.0 * P12_P16)

def getMAR(image, rotateTimes = 0, rotateAngle = 5, rotateDirection = 'clockwise'):
    '''
    计算MAR值
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
    # 左右眼在68个点中的索引
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

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
        
        # 取出嘴巴对应的特征点
        mouth = points[mStart : mEnd]
        # 计算嘴巴的MAR值
        mar = MAR(mouth)

        '''
        # 遍历所有点，打印出其坐标，并圈出来
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 1, (255, 0, 0), 1)
        cv2.imshow("image", img)
        '''

        '''
        # 在图片中标注人脸，并显示
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        # 框出人脸
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 3)
        faceRegion = img[top : bottom, left : right]
        
        # 寻找嘴巴轮廓并绘制出来
        mouthInsideHull = cv2.convexHull(mouth[12 : 20])
        mouthOutsideHull = cv2.convexHull(mouth[0 : 12])
        cv2.drawContours(img, [mouthInsideHull], -1, (255, 0, 0), 1)
        cv2.drawContours(img, [mouthOutsideHull], -1, (255, 0, 0), 1)
        cv2.imshow('i', img)
        cv2.waitKey(100)
        '''

        # 返回一个MAR值
        return mar
    else:
        rotateTimes += 1
        if rotateDirection == 'clockwise':
            if rotateTimes > 5:
                # 顺时针旋转多次仍未检出人脸，改用逆时针旋转
                return getMAR(image, rotateTimes=1, rotateDirection='anticlockwise')
            else:
                # 顺时针旋转再检测
                return getMAR(image, rotateTimes=rotateTimes, rotateDirection='clockwise')
        else:
            if rotateTimes > 5:
                print('No faces detected though rotate for several times, break!')
                return float('inf')
            else:
                # 逆时针旋转再次检测
                return getMAR(image, rotateTimes=rotateTimes, rotateDirection='anticlockwise')

def mouth_detection(videoPath, output_dir, threshold, windowLen):
    '''
    检测嘴巴的张闭状态
    videoPath : 视频地址
    output_dir : 输出视频地址
    threshold : 根据MAR值判断嘴巴是否张开的一个阈值
    windowLen : 滑动窗口长度
    '''
    cap = cv2.VideoCapture(videoPath)

    # 视频编解码器，MPEG-4.2编码类型
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    fps =int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(output_dir, fourcc, fps, size)

    # 记录MAR值
    marList = []
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # 转成灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mar = getMAR(gray)
            marList.append(mar)
            (height, width) = gray.shape
            
            #m = ' %.4f'%(mar)
            m = ''
            # 开始的几帧直接按照跟阈值相比的方式判断是嘴巴的状态是闭合还是张开
            if count < windowLen:
                if mar < threshold:
                    cv2.putText(frame, 'Mouth Closed'+m, (round(0.1 * width), round(0.1 * height)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.FONT_HERSHEY_SIMPLEX)
                else:
                    cv2.putText(frame, 'Mouth Open'+m, (round(0.1 * width), round(0.1 * height)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.FONT_HERSHEY_SIMPLEX)
            else:
                # 检测到嘴巴闭合的情况下需要判断在滑动窗口内是否嘴巴闭合次数占多，是的话才认为是嘴巴闭合，否则仍然认为是嘴巴张开
                if mar < threshold and np.sum(np.array(marList[count - windowLen : ]) < threshold) > windowLen // 2:
                    cv2.putText(frame, 'Mouth Closed'+m, (round(0.1 * width), round(0.1 * height)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.FONT_HERSHEY_SIMPLEX)
                else:
                    cv2.putText(frame, 'Mouth Open'+m, (round(0.1 * width), round(0.1 * height)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.FONT_HERSHEY_SIMPLEX)
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
    print(marList)
    plt.plot(range(len(marList)), marList, 'g-s')
    plt.show()
    '''

def main():
    videoPath = 'Fatigue Detection/mouth'
    output_dir = 'output/mouth'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = os.listdir(videoPath)
    threshold = 0.15
    windowLen = 3

    
    for f in files:
        mouth_detection(os.path.join(videoPath, f), os.path.join(output_dir, f), threshold, windowLen)
    
    '''
    f = files[2]
    mouth_detection(os.path.join(videoPath, f), os.path.join(output_dir, f), threshold, windowLen)
    '''

if __name__ == "__main__":
    main()
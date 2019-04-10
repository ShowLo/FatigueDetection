# -*- coding: UTF-8 -*-

import cv2
import skimage
import imageio
import dlib
import os
import numpy as np
from scipy import stats
import imutils
from imutils import face_utils

def findConnectedComponent(region, threshold, isAdded, index):
    '''
    寻找对应于当前像素的连通分量
    region : 当前区域
    threshold : 判断是否8邻接的一个阈值
    isAdded : 记录当前像素是否已经加入某个连通分量
    index : 当前所处理的像素索引
    connectedComponent : 记录连通分量的元素索引
    '''
    (i, j) = index
    (height, width) = region.shape
    # 把当前像素点加入到连通分量中去并标记已加入
    connectedComponent = [(i, j)]
    isAdded[i][j] = True

    # 建立一个队列存放需要处理的属于当前连通分量的像素信息
    queue = []
    for h in range(max(0, i - 1), min(i + 2, height)):
        for w in range(max(0, j - 1), min(j + 2, width)):
            if not isAdded[h][w] and abs(int(region[h][w]) - int(region[i][j])) < threshold:
                queue.append((h, w, i, j))
    
    # 队列为空的时候说明已经得到了一个连通分量
    while len(queue) > 0:
        # 先进先出
        (h, w, i, j) = queue.pop(0)
        # 8邻域中的像素未加入任何连通分量而且与中心元素的差不超过阈值
        if not isAdded[h][w] and abs(int(region[h][w]) - int(region[i][j])) < threshold:
            # 加入当前连通分量并标记
            isAdded[h][w] = True
            connectedComponent.append((h, w))
            # 然后对其8邻域的元素，只要未加入连通分量的都加入队列中去
            for hi in range(max(0, h - 1), min(h + 2, height)):
                for wj in range(max(0, w - 1), min(w + 2, width)):
                    if not isAdded[hi][wj]:
                        queue.append((hi, wj, h, w))
    return connectedComponent

def findMaxConnectedComponent(region, minValue, maxValue, connectThreshold):
    '''
    寻找属于与脸部肤色相近的面积最大的连通分量（8连通）
    region : 进行查找的区域
    minValue, maxValue : 脸部肤色的大致范围
    connectThreshold : 判断两个像素是否8连通的阈值
    '''
    # 将灰度值较小的置零
    region[region < minValue] = 0
    region[region > maxValue] = 0
    (height, width) = region.shape
    # 记录是否已经加入某个连通分量
    isAdded = [[False for j in range(width)] for i in range(height)]
    allConnectedComponent = []
    for i in range(height):
        for j in range(width):
            # 如果与脸部肤色相近且未加入某个连通分量，则以当前像素为基准寻找一个连通分量
            if not isAdded[i][j] and (region[i][j] >= minValue and region[i][j] <= maxValue):
                connectedComponent = findConnectedComponent(region, connectThreshold, isAdded, (i, j))
                if connectedComponent:
                    allConnectedComponent.append(connectedComponent)
    # 根据连通分量中元素个数的多少进行排序
    allConnectedComponent.sort(key = lambda x : len(x))
    # 返回元素个数最多的连通分量
    if len(allConnectedComponent) > 0:
        return allConnectedComponent[-1]
    else:
        return None

def detectHand(image, faceRegionLRTB, connectThreshold):
    '''
    检测可能存在于人脸左右的手
    image : 输入灰度图像
    faceRegionLRTB : 人脸范围，顺序为左右上下
    connectThreshold : 判断邻接与否的阈值
    '''
    (height, width) = image.shape
    (left, right, top, bottom) = faceRegionLRTB
    # 额外增加高度，防止手的位置过低
    extraHeight = (bottom - top) // 2
    # 人脸的左右区域(以图片中人的方位为准)
    leftRegion = image[top : min(height, bottom + extraHeight), right :]
    rightRegion = image[top : min(height, bottom + extraHeight), 0 : left]
    faceRegion = image[top : bottom, left : right]
    # 以脸部的灰度值最大值的三分之一作为最小值
    minValue = int(np.max(faceRegion) / 3)
    maxValue = int(np.max(faceRegion))
    
    # 分别检测人脸左边和右边区域可能存在的手
    leftHand = findMaxConnectedComponent(leftRegion, minValue, maxValue, connectThreshold)
    rightHand = findMaxConnectedComponent(rightRegion, minValue, maxValue, connectThreshold)

    # 返回两者中面积较大者
    if leftHand or rightHand:
        if leftHand is None:
            return rightHand
        elif rightHand is None:
            return leftHand
        else:
            if len(leftHand) > len(rightHand):
                return leftHand
            else:
                return rightHand
    else:
        return None

def detectCalling(image, rotateTimes = 0, rotateAngle = 5, rotateDirection = 'clockwise'):
    '''
    计算手部与脸部面积比
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
        
        # 在图片中标注人脸，并显示
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        # 框出人脸
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 3)
        faceRegion = img[top : bottom, left : right]
        faceArea = np.sum(faceRegion > np.max(faceRegion) / 3)

        # 判断是否邻接的阈值
        connectThreshold = 4
        # 检测可能的手部
        possibleHand = detectHand(img, (left, right, top, bottom), connectThreshold)
        handArea = len(possibleHand)
        
        '''
        for (i, j) in possibleHand:
            img[i+top][j] = 255

        
        cv2.imshow('i', img)
        cv2.waitKey(100)
        '''
        

        # 返回
        return handArea / faceArea
    else:
        rotateTimes += 1
        if rotateDirection == 'clockwise':
            if rotateTimes > 5:
                # 顺时针旋转多次仍未检出人脸，改用逆时针旋转
                return detectCalling(image, rotateTimes=1, rotateDirection='anticlockwise')
            else:
                # 顺时针旋转再检测
                return detectCalling(image, rotateTimes=rotateTimes, rotateDirection='clockwise')
        else:
            if rotateTimes > 5:
                print('No faces detected though rotate for several times, break!')
                return float('inf')
            else:
                # 逆时针旋转再次检测
                return detectCalling(image, rotateTimes=rotateTimes, rotateDirection='anticlockwise')

def calling_detection(videoPath, output_dir, handFaceRatioThreshold, windowLen):
    '''
    检测是否打电话
    videoPath : 视频地址
    output_dir : 输出视频地址
    handFaceRatioThreshold : 根据手与脸的面积比判断手部是否存在的阈值
    windowLen : 滑动窗口长度
    '''
    cap = cv2.VideoCapture(videoPath)

    # 视频编解码器，MPEG-4.2编码类型
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    fps =int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(output_dir, fourcc, fps, size)

    # 记录面积比
    handFaceRationList = []
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # 转成灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 计算手部与面部面积比
            handFaceRation = detectCalling(gray)
            handFaceRationList.append(handFaceRation)
            (height, width) = gray.shape
            
            #m = ' %.4f'%(handFaceRation)
            m = ''
            # 开始的几帧直接按照跟阈值相比的方式判断是否打电话中
            if count < windowLen:
                if handFaceRation < handFaceRatioThreshold:
                    cv2.putText(frame, 'Normal'+m, (round(0.1 * width), round(0.1 * height)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.FONT_HERSHEY_SIMPLEX)
                else:
                    cv2.putText(frame, 'Calling'+m, (round(0.1 * width), round(0.1 * height)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.FONT_HERSHEY_SIMPLEX)
            else:
                # 检测到正常状态的情况下需要判断在滑动窗口内是否正常状态占多，是的话才认为在正常状态，否则仍然认为是在打电话
                if handFaceRation < handFaceRatioThreshold and np.sum(np.array(handFaceRationList[count - windowLen : ]) < handFaceRatioThreshold) > windowLen // 2:
                    cv2.putText(frame, 'Normal'+m, (round(0.1 * width), round(0.1 * height)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.FONT_HERSHEY_SIMPLEX)
                else:
                    cv2.putText(frame, 'Calling'+m, (round(0.1 * width), round(0.1 * height)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.FONT_HERSHEY_SIMPLEX)
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
    print(handFaceRationList)
    plt.plot(range(len(handFaceRationList)), handFaceRationList, 'g-s')
    plt.show()
    '''
    

def main():
    videoPath = 'Fatigue Detection/calling'
    output_dir = 'output/calling'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = os.listdir(videoPath)
    threshold1 = 0.38
    threshold2 = 0.24
    threshold3 = 0.3
    windowLen = 3

    threshold = [0.5, 0.45, 0.2, 0.43, threshold1, threshold1, threshold3, threshold2, 0.14, threshold2, threshold2, threshold2, threshold2, threshold2]

    
    for i in range(len(files)):
        calling_detection(os.path.join(videoPath, files[i]), os.path.join(output_dir, files[i]), threshold[i], windowLen)
    
    
    '''
    f = files[13]
    calling_detection(os.path.join(videoPath, f), os.path.join(output_dir, f), threshold2, windowLen)
    '''
    

if __name__ == "__main__":
    main()
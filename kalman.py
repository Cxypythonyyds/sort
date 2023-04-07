from __future__ import print_function
from numba import jit
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

@jit
def iou(bb_test,bb_gt):     #求交并比
    # 左上角坐标的最大值
    xx1 = np.maximum(bb_test[0],bb_gt[0])
    yy1 = np.maximum(bb_test[1],bb_gt[1])
    # 右下角坐标
    xx2 = np.minimum(bb_test[2],bb_gt[2])
    yy2 = np.minimum(bb_test[3],bb_gt[3])
    #交的宽高
    w = np.maximum(0,xx2-xx1)
    h = np.maximum(0,yy2-yy1)
    #交的面积
    wh = w*h
    #并的面积
    s = (bb_test[2]-bb_test[0]*(bb_test[3]-bb_test[1])+(bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]-wh))
    return wh / s


# 将左上角坐标[x1，y1，x2，y2]转换成[x,y,s,r]
def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2
    y = bbox[1] + h/2
    s = w*h
    r = w/float(h) #r：宽高比
    return np.array[x,y,s,r].reshape((4,1))

#将[x,y,s,r]装换成[x1,y1,x2,y2]
def convert_x_to_bbox(x,score = None): #score:置信度
    w = np.sqrt(x[2]*x[3]) #sqrt开根号
    h = x[2]/w  #面积÷宽 = 高
    x1 = x[0] - w/2
    y1 = x[1] - h/2
    x2 = x[0] + w/2
    y2 = x[1] + h/2
    if score is None:
        return np.array[x1,y1,x2,y2].reshape((1,4))
    else:
        return np.array[x1,y1,x2,y2,score]



class KalmanBoxTracker(object):
    count = 0

def __init__(self, bbox):                           # bbox获取的是yolo的检测结果

    # 初始化边界框和跟踪器
    # :param bbox:

    # 定义等速模型
    # 内部使⽤KalmanFilter，7个状态变量和4个观测输⼊
    self.kf = KalmanFilter(dim_x=7, dim_z=4)

    self.kf.F = np.array(                           #F：状态转移矩阵
        [[1, 0, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 0, 1],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1]])
    self.kf.H = np.array(                           #H：测量矩阵
        [[1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]])
    self.kf.R[2:, 2:] *= 10.                        #R:测量噪声协方差，单位矩阵后面从2开始都乘10
    self.kf.P[4:, 4:] *= 1000.                      #P：先验估计协方差 give high uncertainty to the unobservable
    self.kf.P *= 10.
    self.kf.Q[-1, -1] *= 0.01
    self.kf.Q[4:, 4:] *= 0.01
    self.kf.x[:4] = convert_bbox_to_z(bbox)         #X：观测结果 (u,v,s,r)
    self.time_since_update = 0                      #更新后预测的次数
    self.id = KalmanBoxTracker.count                #卡尔曼滤波器的个数
    KalmanBoxTracker.count += 1                     #检测到加一
    self.history = []                               #跟踪到的所有目标放入这里
    self.hits = 0                                   #跟踪到目标的次数
    self.hit_streak = 0                             #跟踪到目标的次数
    self.age = 0                                    #跟踪目标存在的帧数


def update(self, bbox):
                                                    # 使⽤观察到的⽬标框更新状态向量。filterpy.kalman.KalmanFilter.update 会根据观测
                                                    # 重置self.time_since_update，清空self.history。
                                                    # :param bbox:⽬标框
                                                    # :return:
    self.time_since_update = 0                      #重置
    self.history = []                               #清空
    self.hits += 1                                  #hits计数加1，，，表示整个目标在视频中存在的帧数
    self.hit_streak += 1                            #当time_since_update不等于0时hit_streak加一
    self.kf.update(convert_bbox_to_z(bbox))         #根据观测结果修改内部状态x


def predict(self):
    # 推进状态向量并返回预测的边界框估计。
    # 将预测结果追加到self.history。由于 get_state 直接访问 self.kf.x，所以self.his
    # :return
    if (self.kf.x[6] + self.kf.x[2]) <= 0:              #推进状态变量
        self.kf.x[6] *= 0.0
    self.kf.predict()                                    #进⾏预测，，，卡尔曼滤波的次数
    self.age += 1                                        #只要预测就加一，目标在视频中存在的帧数
    if self.time_since_update > 0:                       #若过程中未更新过，将hit_streak置为0
        self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))    #将预测结果追加到history中
    return self.history[-1]

def get_state(self):
    # 返回当前边界框估计值
    # :return:
    return convert_x_to_bbox(self.kf.x)

#将yolo模型的检测框和卡尔曼滤波的跟踪目标框进行匹配
#可以对这部分进行更改：iou改成特征向量

def  associate_detections_to_trackers(detections,trackers,iou_threshold=0.3):
    # 返回：
    # 跟踪跟踪成功目标
    # 跟踪失败的的目标
    #跟踪或检测为0，直接构造返回结果
    if len(trackers) == 0 or (len(detections) == 0):
        return np.empty((0,2),dtype=int),np.arange(len(detections)),np.empty((0,5),dtype=int)

    #IOU逐个进行交并比的计算，构造矩阵，sciyp linerar_assignment进行匹配（匈牙利算法）
    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)  #改成特征向量可以对这部分加上欧氏距离进行更改
    #遍历检测框(把每一个检测框存入)
    for d,det in enumerate(detections):
                                                     #eg:
                                                     #   i = 0
                                                     #   seq = ['one', 'two', 'three']
                                                     #   for i, element in enumerate(seq):
                                                     #   print i，element
                                                     #   i += 1
                                                     #out:
                                                     #     0 one
                                                     #     2 two
                                                     #     3 three
        #遍历跟踪框
        for t,trk in enumerate(trackers):            #整个遍历相当于一个检测框分别和跟踪框进行存入iou_matrix [1]
            iou_matrix[d,t] = iou(det,trk)           #遍历之后的检测框和跟踪框的 交并比 存入 d,t相当于x,y,相当于记录元素的位置
                                                     #iou_matrix[d,t]相当于存的是每当一个检测框对应的一个跟踪框时记录
                                                     #（假设第一个检测框）矩阵0，0的位置在对应下一个跟踪框时则是0，1
    result = linear_sum_assignment(-iou_matrix)      #调用liner_assignments 进行匹配    加负号是为了得到最大匹配结果，不加则是最小匹配结果
    matched_indices = np.array(list(zip(*result)))   #将匹配结果以[[0,0][0,1].......]列表展示,,,第0个检测检测第0个跟踪的坐标，第0个检测检测第1个跟踪的坐标
    unmatched_detections = []                        #记录未匹配的结果     未匹配的检测框：新的目标进入到画面中，更新新增结果
    for d,det in enumerate(detections):
        if d not in matched_indices[:,0]:
            unmatched_detections.append(d)
    unmatched_tracker = []                           #未匹配的跟踪框：目标离开了画面
    for t,trk in enumerate(trackers):
        if t not in matched_indices[:,1]:
            unmatched_tracker.append(t)
    for m in matched_indices:                        #将匹配成功的跟踪框放入
        if iou_matrix[m[0],m[1]]<iou_threshold:      #m[0]表示detection的坐标m[1]表示track的坐标，iou_matrix(m[0]，m[1])表示第m[0]个检测第m[1]个跟踪时的iou值
            unmatched_detections.append(m[0])
            unmatched_tracker.append(m[1])
        else:
            matchs = np.empty(m.reshape(1,2) )       #列向量输出
        # 格式转换
        if len(matchs) == 0:                         #没有匹配到结果，这里感觉可以处理
            matchs = np.empty((0,2),dtype = int)
        else:                                        #匹配成功时返回匹配的结果
            matchs = np.concatenate(matchs,axis = 0)
        return matchs,np.array(unmatched_detections),np.array(unmatched_tracker)



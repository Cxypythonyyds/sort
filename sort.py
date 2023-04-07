from __future__ import print_function
from numba import jit
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from kalman import KalmanBoxTracker,associate_detections_to_trackers


def __init__(self,max_age, min_hits = 3):
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
def update(self, dets):
    self.frame_count += 1
 # 在当前帧逐个预测轨迹位置，记录状态异常地跟踪器索引
 # 根据当前所有的卡尔曼跟踪器个数（即上⼀帧中跟踪的⽬标个数）创建⼆维数组：⾏号为
    trks = np.zeros((len(self.trackers), 5)) # 存储跟踪器的预测
    to_del = [] # 存储要删除的⽬标框
    ret = [] # 存储要返回的追踪⽬标框
 # 循环遍历卡尔曼跟踪器列表
    for t, trk in enumerate(trks):
 # 使⽤卡尔曼跟踪器t产⽣对应⽬标的跟踪框
        pos = self.trackers[t].predict()[0]
 # 遍历完成后，trk中存储了上⼀帧中跟踪的⽬标的预测跟踪框
        trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
 # 如果跟踪框中包含空值则将该跟踪框添加到要删除的列表中
        if np.any(np.isnan(pos)):
            to_del.append(t)
 # numpy.ma.masked_invalid 屏蔽出现⽆效值的数组（NaN 或 inf）
 # numpy.ma.compress_rows 压缩包含掩码值的2-D 数组的整⾏，将包含掩码值的整⾏
 # trks中存储了上⼀帧中跟踪的⽬标并且在当前帧中的预测跟踪框
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
 # 逆向删除异常跟踪器，防⽌破坏索引
    for t in reversed(to_del):
        self.trackers.pop(t)
 # 将⽬标检测框与卡尔曼滤波器预测的跟踪框关联获取跟踪成功的⽬标，新增的⽬标，离开画面目标
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)
 # 将跟踪成功的⽬标框更新到对应的卡尔曼滤波器
    for t, trk in enumerate(self.trackers):
        if t not in unmatched_trks:
            d = matched[np.where(matched[:, 1] == t)[0], 0]
 # 使⽤观测的边界框更新状态向量
            trk.update(dets[d, :][0])
 # 为新增的⽬标创建新的卡尔曼滤波器对象进⾏跟踪
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i, :])
        self.trackers.append(trk)
 # ⾃后向前遍历，仅返回在当前帧出现且命中周期⼤于self.min_hits（除⾮跟踪刚开始)的跟踪结果；如果未命中时间大于self.max_age则删除跟踪器
 # hit_streak忽略⽬标初始的若⼲帧
    i = len(self.trackers)
    for trk in reversed(self.trackers):
 # 返回当前边界框的估计值
        d = trk.get_state()[0]
 # 跟踪成功⽬标的box与id放⼊ret列表中
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
            ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))    #+1 as MOT benchmark requires positive
        i -= 1
 # 跟踪失败或离开画⾯的⽬标从卡尔曼跟踪器中删除
    if trk.time_since_update > self.max_age:
        self.trackers.pop(i)
 # 返回当前画⾯中所有⽬标的box与id,以⼆维矩阵形式返回
    if len(ret) > 0:
        return np.concatenate(ret)
    return np.empty((0, 5))

# @Time    : 2018/12/4 14:57
# @Email  : wangchengo@126.com
# @File   : STMatrix.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0
import os, sys
sys.path.append('../../')

import numpy as np
import pandas as pd
from data.TaxiBJ.timestamp import string2timestamp


class STMatrix(object):
    """docstring for STMatrix"""

    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps  # [b'2013070101', b'2013070102']
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()  # 将时间戳：做成一个字典，也就是给每个时间戳一个序号

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i - 1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):  # 给定时间戳返回对于的数据
        return self.data[self.get_index[timestamp]]

    def save(self, fname):
        pass

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset(self, 
                       len_closeness=4, 
                       len_period=0, PeriodInterval=1, 
                       len_trend=0, TrendInterval=7, 
                       len_horizon=4):

        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)  # 时间偏移 minutes = 30
        
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness + 1),
                   [PeriodInterval * self.T * j for j in range(1, len_period + 1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend + 1)],]
        # print depends # [range(1, 4), [48, 96, 144], [336, 672, 1008]]
        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)

        while i < len(self.pd_timestamps):
            
            Flag = True
            for depend in depends:
                if Flag is False: break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])
            
            Flag &= self.check_it([self.pd_timestamps[i] + j * offset_frame for j in range(len_horizon)])                

            if Flag is False:
                i += 1
                continue
            
            """ get the direct neighbors 
            E.g., [Timestamp('2013-07-01 00:00:00')] =>
            [Timestamp('2013-06-30 23:30:00'), Timestamp('2013-06-30 23:00:00'), Timestamp('2013-06-30 22:30:00')]
            
            """
            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
            
            """ get the daily neighbors 
            E.g., if len_period=3, get the value from previous 1,2,3 days at the same time
            
            """
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]
            
            """ get the week neighbors 
            E.g., if len_trend=3, get the value from previous 7, 14, 21 days at the same time
            
            """
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]

            """ get forecast ground truth """
            y = [self.get_matrix(self.pd_timestamps[i] + j * offset_frame) for j in range(len_horizon)]
            
            if len_closeness > 0:
                XC.append(np.stack(x_c))
                # a.shape=[2,32,32] b.shape=[2,32,32] c=np.vstack((a,b)) -->c.shape = [2,2,32,32]
            if len_period > 0:
                XP.append(np.stack(x_p))
            if len_trend > 0:
                XT.append(np.stack(x_t))
                
            Y.append(np.stack(y))
            timestamps_Y.append(self.timestamps[i])
            i += 1
            
        XC = np.asarray(XC)  # closeness [?, 4, 2, 32, 32]
        XP = np.asarray(XP)  # daily
        XT = np.asarray(XT)  # weekly
        Y = np.asarray(Y)    # output [?, 4, 2, 32, 32]
        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y, timestamps_Y


if __name__ == '__main__':
    # depends = [range(1, 3 + 1),
    #            [1 * 48 * j for j in range(1, 3 + 1)],
    #            [7 * 48 * j for j in range(1, 3 + 1)]]
    # print(depends)
    # print([j for j in depends[0]])
    str = ['2013070101']
    t = string2timestamp(str)
    offset_frame = pd.DateOffset(minutes=24 * 60 // 48)  # 时间偏移 minutes = 30
    print(t)
    o = [t[0] - j * offset_frame for j in range(1, 4)]
    print(o)

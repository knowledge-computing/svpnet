# -*- coding: utf-8 -*-
"""
    load BJ Data from multiple sources as follows:
        meteorologic data
"""
from __future__ import print_function

import os, sys
sys.path.append('../../')

import time
import pickle
from copy import copy
import numpy as np
import h5py

from data.TaxiBJ.STMatrix import STMatrix
from data.TaxiBJ.timestamp import timestamp2vec
from data.TaxiBJ.MaxMinNormalization import MinMaxNormalization


def load_holiday(data_path, timeslots):
    """
    load holiday data
    return:
        [[1],[1],[0],[0],[0]...], where 1 is the holiday, 0 is the non-holiday
    """
    
    fname=os.path.join(data_path, 'BJ_Holiday.txt')
    f = open(fname, 'r')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])
    H = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        if slot[:8] in holidays:
            H[i] = 1
    # print(timeslots[H==1])
    return H[:, None]  # into 2 dims


def load_meteorol(data_path, timeslots):
    '''
    timeslots: the predicted timeslots
    In real-world, we dont have the meteorol data in the predicted timeslot, 
    Instead, we use the meteoral at previous timeslots, i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)

    '''
    
    fname=os.path.join(data_path, 'BJ_Meteorology.h5')
    f = h5py.File(fname, 'r')
    Timeslot = list(f['date'])
    WindSpeed = np.array(f['WindSpeed'])
    Weather = np.array(f['Weather'])
    Temperature = np.array(f['Temperature'])
    f.close()

    M = dict()  # map timeslot to index
    for i, slot in enumerate(Timeslot):
        M[slot] = i

    WS = []  # WindSpeed
    WR = []  # Weather
    TE = []  # Temperature
    for slot in timeslots:
        predicted_id = M[slot]
        cur_id = predicted_id - 1
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)

    # 0-1 scale
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())

    print("meteorol shape: ", 'WindSpeed', WS.shape, "Weather", WR.shape, 'Temperature', TE.shape)

    # concatenate all these attributes
    merge_data = np.hstack([WR, WS[:, None], TE[:, None]])

    # print('meger shape:', merge_data.shape)
    return merge_data


def load_stdata(fname):
    """
    split the data and date(timestamps)
    :param fname:
    :return:
    """
    f = h5py.File(fname, 'r')
    data = np.array(f['data'])
    timestamps = list(f['date'])
    f.close()
    return data, timestamps


def stat(fname):
    """
    count the valid data
    :param fname:
    :return: like below

    ==========stat==========
    data shape: (7220, 2, 32, 32)
    # of days: 162, from 2015-11-01 to 2016-04-10
    # of timeslots: 7776
    # of timeslots (available): 7220
    missing ratio of timeslots: 7.2%
    max: 1250.000, min: 0.000
    ==========stat==========

    """

    def get_nb_timeslot(f):
        """
        count the number of timeslot of given data
        :param f:
        :return:
        """
        s = f['date'][0]
        e = f['date'][-1]
        year, month, day = map(int, [s[:4], s[4:6], s[6:8]])
        ts = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        year, month, day = map(int, [e[:4], e[4:6], e[6:8]])
        te = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        nb_timeslot = (time.mktime(te) - time.mktime(ts)) / (0.5 * 3600) + 48
        time_s_str, time_e_str = time.strftime("%Y-%m-%d", ts), time.strftime("%Y-%m-%d", te)
        return nb_timeslot, time_s_str, time_e_str

    with h5py.File(fname) as f:
        nb_timeslot, time_s_str, time_e_str = get_nb_timeslot(f)
        nb_day = int(nb_timeslot / 48)
        mmax = np.array(f['data']).max()
        mmin = np.array(f['data']).min()
        stat = '=' * 10 + 'stat' + '=' * 10 + '\n' + \
               'data shape: %s\n' % str(f['data'].shape) + \
               '# of days: %i, from %s to %s\n' % (nb_day, time_s_str, time_e_str) + \
               '# of timeslots: %i\n' % int(nb_timeslot) + \
               '# of timeslots (available): %i\n' % f['date'].shape[0] + \
               'missing ratio of timeslots: %.1f%%\n' % ((1. - float(f['date'].shape[0] / nb_timeslot)) * 100) + \
               'max: %.3f, min: %.3f\n' % (mmax, mmin) + \
               '=' * 10 + 'stat' + '=' * 10
        print(stat)


def remove_incomplete_days(data, timestamps, T=48):
    """
    remove a certain day which has not 48 timestamps
    :param data:
    :param timestamps:
    :param T:
    :return:
    """

    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i + T - 1 < len(timestamps) and int(timestamps[i + T - 1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps


def load_dataset(data_path, T=48, nb_flow=2, 
                 len_closeness=0, len_period=0, len_trend=0, len_horizon=4, 
                 preprocess_name='preprocessing.pkl',
                 meta_data=False, meteorol_data=False, holiday_data=False):
    """
    load the preprocessed dataset
    :param T:
    :param nb_flow:
    :param len_closeness:
    :param len_period:
    :param len_trend:
    :param len_test:
    :param preprocess_name:
    :param meta_data:
    :param meteorol_data:
    :param holiday_data:
    :return:
    """
    assert (len_closeness + len_period + len_trend > 0)
    # load data
    # 13 - 16
    data_all = []
    timestamps_all = list()
    for year in range(13, 17):
        fname = os.path.join(data_path, 'BJ{}_M32x32_T30_InOut.h5'.format(year))
        print("file name: ", fname)
        # stat(fname)
        data, timestamps = load_stdata(fname)
        # print(timestamps)
        # remove a certain day which does not have 48 timestamps
        data, timestamps = remove_incomplete_days(data, timestamps, T)
        data = data[:, :nb_flow]
        data[data < 0] = 0.
        data_all.append(data)
        timestamps_all.append(timestamps)
        print("\n")

    # minmax_scale
    data_train = np.vstack(copy(data_all))
    print('train_data shape: ', data_train.shape)

    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = [mmn.transform(d) for d in data_all]
    
    fpkl = open(os.path.join(data_path, 'CACHE', preprocess_name), 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)  # 保存特征缩放模型[-1,1]
    fpkl.close()
    # print(len(data_all_mmn[0]))
    # print(timestamps_all[0][:10])
    
    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is
        # a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(len_closeness=len_closeness, 
                                                             len_period=len_period, 
                                                             len_trend=len_trend, 
                                                             len_horizon=len_horizon)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y  # [ b'2013102232', b'2013102233', b'2013102234', b'2013102235',......]
        
    meta_feature = []
    if meta_data:
        # load time feature
        time_feature = timestamp2vec(timestamps_Y)  # array: [?,8]
        meta_feature.append(time_feature)
        
    if holiday_data:
        # load holiday
        holiday_feature = load_holiday(data_path, timestamps_Y)
        meta_feature.append(holiday_feature)
        
    if meteorol_data:
        # load meteorol data
        meteorol_feature = load_meteorol(data_path, timestamps_Y)
        meta_feature.append(meteorol_feature)

    meta_feature = np.hstack(meta_feature) if len(
        meta_feature) > 0 else np.asarray(meta_feature)
    
    metadata_dim = meta_feature.shape[1] if len(
        meta_feature.shape) > 1 else None
    
    if metadata_dim is not None and metadata_dim < 1:
        metadata_dim = None
    
    if meta_data and holiday_data and meteorol_data:
        print('time feature:', time_feature.shape, 'holiday feature:', holiday_feature.shape,
              'meteorol feature: ', meteorol_feature.shape, 'mete feature: ', meta_feature.shape)

    XC = np.vstack(XC)  # shape = [15072,4,2,32,32]
    XP = np.vstack(XP)  # shape = [15072,2,32,32]
    XT = np.vstack(XT)  # shape = [15072,2,32,32]
    Y = np.vstack(Y)  # shape = [15072,2,32,32]

    print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, 
          "Y shape:", Y.shape, "meta_feature", meta_feature.shape)
        
    XP = XP if XP.shape[1] > 0 else None
    XT = XT if XT.shape[1] > 0 else None
    meta_feature = meta_feature if meta_feature.shape[0] > 0 else None

    
    return {
        'XC': XC,
        'XP': XP,
        'XT': XT,
        'Y': Y,
        'XM': meta_feature,
        'T': timestamps_Y,
    }
#     XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
#     XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
#     timestamp_train, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[-len_test:]
    
#     X_train, X_test = [], []

#     for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
#         if l > 0:
#             X_train.append(X_)
#     for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
#         if l > 0:
#             X_test.append(X_)
#     print('XC_train shape:', XC_train.shape, Y_train.shape, 'XC_test shape: ', XC_test.shape, Y_test.shape)
    
#     print(meta_feature.shape)
    
#     import sys
#     sys.exit(1)
    
#     if metadata_dim is not None:
#         meta_feature_train, meta_feature_test = meta_feature[:-len_test], meta_feature[-len_test:]
#         X_train.append(meta_feature_train)
#         X_test.append(meta_feature_test)

#     for _X in X_train:
#         print(_X.shape, )
#     print()
#     for _X in X_test:
#         print(_X.shape, )
#     print()
#     return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test


def cache(fname, dataset):
    
    h5 = h5py.File(fname, 'w')
    for k, v in dataset.items():
        if v is not None:
            h5.create_dataset(k, data=v)
    h5.close()


def read_cache(fname):
    
    f = h5py.File(fname, 'r')
    
    dataset = dict()
    for k in f.keys():
        if k != 'T':
            dataset[k] = np.array(f[k])
        else:
            dataset[k] = list(f[k])
    f.close()
    return dataset


def load_data(data_path, 
              len_closeness=4, len_period=0, len_trend=0, len_horizon=4, 
              meta_data=False, meteorol_data=False, holiday_data=False):
    
    cache_path = os.path.join(data_path, 'CACHE')
    fname = os.path.join(cache_path, 'TaxiBJ_C{}_P{}_T{}_H{}.h5'.format(len_closeness, len_period, len_trend, len_horizon))
    
    if os.path.exists(fname):
        dataset = read_cache(fname)
        print("load %s successfully" % fname)
        
    else:
        
        if os.path.isdir(cache_path) is False:
            os.mkdir(cache_path)
            
        dataset = \
            load_dataset(data_path, 
                         len_closeness=len_closeness, 
                         len_period=len_period, 
                         len_trend=len_trend, 
                         len_horizon=len_horizon,
                         meta_data=meta_data, 
                         meteorol_data=meteorol_data, 
                         holiday_data=holiday_data)
        
        cache(fname, dataset)
        
    return dataset  # {'XC','XP','XT','Y','XM','Timestamps'}


if __name__ == "__main__":
    # load_data(T=48, nb_flow=2, len_closeness=3, len_period=1, len_trend=1, len_test=48 * 28)
    # print(DATAPATH)
    # print(CACHEPATH)
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = \
        load_data(len_closeness=3, len_period=1, len_trend=1, len_test=28 * 48)

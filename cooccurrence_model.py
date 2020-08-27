#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os 
import re
import glob
import spark
import pandas as pd
import functools
import json
import time
import datetime
import numpy as np
from collections import defaultdict
from datetime import timedelta


# In[3]:


'''
This function calculate the cooccurrence count within a single timespan.
There are two groups to maintain the state, left group records device ids from previous time window,
right group records new incoming device ids.
There are three conditions to update the cooccurrence pairs:
1. left device is from left group, right device is from and only exists in right group without occur in left device's timespan 
2. left device is from right group, right device is from and only exists in left group.
3. both devices are from right group
'''
def count_cooccurrence(l, r, rr, cooccurrence, obsolete_index, devices):
    """
    l: left index of left group
    """
    left_freq = defaultdict(int)
    right_freq = defaultdict(int)
    for i in devices[l: r]:
        left_freq[i] += 1
    for i in devices[r: rr]:
        right_freq[i] += 1
        
    rd = right_freq.keys()
    ld = left_freq.keys()
    prev = -1
    obsolete_devices = set()
    # condition 1
    for i in rd: 
        if (not i in ld):
            for k in range(l, r):
                j = devices[k]
                st = obsolete_index[k]
                if prev != st:
                    obsolete_devices = set(devices[st: k]) 
                    prev = st
                if not i in obsolete_devices: 
                    cooccurrence[j, i] += 1
    # condition 2               
    for i in rd: 
        for j in ld:
            if not j in rd: 
                cooccurrence[i, j] += right_freq[i]
    
    # condition 3
    for i in rd:
        for j in rd: 
            if i != j:
                cooccurrence[i, j] += right_freq[i]    
                
"""
This function slides the time window from left to right in steps of timestamps,
every time left index will try to shift one timestamp to right direction, and right index will find the 
right most index within the time window starts from left index.
It will also maintain every device left most index within time window
"""                
def create_context(timespan, df, cooccurrence, device_vocab):
    df = df.sort_values('localTimestamp')
    timestamps = df.localTimestamp.tolist()
    types = df.type.tolist()
    devices = [device_vocab[mac] for mac in df.deviceMac.tolist()]
    l, r = 0, 0 
    obsolete_index = []
    while True:
        ll = l
        while ll < r:
            assert timestamps[ll] >= lt
            if timestamps[ll] != lt:
                break
            ll += 1
        if ll == len(timestamps):
            break
        l = ll
        lt = timestamps[l]
        rr = r
        while (rr < len(timestamps)) and (timestamps[rr] < lt + timedelta(minutes=timespan)):
            rr += 1
        if r == rr: continue
        obsolete_index.extend([l]*(rr-r))
        count_cooccurrence(l, r, rr, cooccurrence, obsolete_index, devices)
        r = rr
    assert len(obsolete_index) == len(devices)
        
def occurrence_model(daily_ap_df, timespan, group_columns=['apMac', 'floorNumber']):
    # create device to id mapping
    device_vocab = {mac: i for i, mac in enumerate(daily_ap_df.deviceMac.unique().tolist())}
    n = len(device_vocab)
    cooccurrence = np.zeros((n, n))
    device_freq = np.zeros((n, 1))
    for mac in daily_ap_df.deviceMac.tolist():
        device_freq[device_vocab[mac]] += 1

    for _, df in daily_ap_df.groupby(group_columns):
        create_context(timespan, df, cooccurrence, device_vocab)
    return cooccurrence / device_freq, device_vocab, device_freq


# In[4]:


# Union find
class Union:
    def __init__(self, n):
        self.par = [-1]*n
        
    def find(self, x):
        par = self.par
        if par[x] < 0:
            return x
        else:
            par[x] = self.find(par[x])
            return par[x]

    def unite(self, x, y):
        par = self.par
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return False
        else:
            if par[x] > par[y]:
                x,y = y,x
            par[x] += par[y]
            par[y] = x
            return True

    def same(x,y):
        return self.find(x) == self.find(y)

    def size(x):
        return -self.par[self.find(x)]

def device_count(volatile_devices, uf, device_vocab, df):
    badge_devices = set()
    for device_mac in df.deviceMac.unique().tolist():
        if not device_mac in device_vocab:
            continue
        device_id = device_vocab[device_mac]
        if device_id in volatile_devices:
            continue
        parent_id = uf.find(device_id)
        if not parent_id in badge_devices:
            badge_devices.add(parent_id)
        else:
            k2 += 1
    return len(badge_devices)

def simulation(daily_ap_df, badge_day_count, timespan, disjoint_overlap, joint_overlap, freq_thresh):
    model_prob, device_vocab, _  = occurrence_model(daily_ap_df, timespan)
    ap_count_df = daily_ap_df.groupby('deviceMac').count()
    day_n = len(daily_ap_df.date.unique())
    # filter out noisy devices
    visitor_group = ap_count_df[ap_count_df.buildingUUID < freq_thresh]
    daily_ap_df = daily_ap_df[~daily_ap_df.deviceMac.isin(visitor_group.index)]
    # filter out devices that have no connection to others
    model_prob_sum = np.sum(model_prob > disjoint_overlap, axis=1)
    volatile_devices = np.where(model_prob_sum == 0)[0]
    # use union find to group high cooccurrent devices into single device 
    n = len(model_prob)
    uf = Union(n)
    undirect_graph = set()
    row, col = np.where(model_prob > joint_overlap)
    for x, y in zip(row, col):
        undirect_graph.add((x, y))
    k = 0
    for (x, y) in undirect_graph:
        if (y, x) in undirect_graph:
            uf.unite(x, y) 
            k += 1
    partial_device_count = functools.partial(device_count, volatile_devices, uf, device_vocab)
    wifi_count = daily_ap_df.groupby('date').apply(partial_device_count)
    wifi_count = wifi_count.to_frame()
    wifi_day_count = wifi_count.rename(columns={0: 'wifi_count'}).reset_index(drop=False)
    wifi_day_count['date'] = pd.to_datetime(wifi_day_count['date'])
    
    # evaluation
    wifi_and_badge = wifi_day_count.merge(badge_day_count, on=['date'])
    wifi_and_badge['perc_error'] = (wifi_and_badge.wifi_count - wifi_and_badge.value.apply(float)).abs() / wifi_and_badge.value.apply(float)
    return wifi_and_badge.perc_error.mean() * 100


# In[5]:


from bayes_opt import BayesianOptimization    
from bayes_opt import SequentialDomainReductionTransformer

black_box_function = functools.partial(simulation, daily_ap_df, badge_day_count)
pbounds = {'timespan': (0, 30), 'disjoint_overlap': (.2, .5), 'joint_overlap': (.5, .9), 'freq_thresh': (10, 30)}
bounds_transformer = SequentialDomainReductionTransformer()

mutating_optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=1,
    random_state=1,
    bounds_transformer=bounds_transformer
)


# In[ ]:





# In[ ]:





import csv
import copy
import functools
import glob
import os

from collections import namedtuple

import SimpleITK as sitk

import numpy as np

import torch
import torch.cuda
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('part2ch10_raw')


CandidateInforTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)

@functools.lru_cache(1) # standard library in-memory caching
def getCandidateInfoList(requireOnDisk_bool=True):
    '''
    파싱이 느릴 수 있기 때문에 함수의 결과를 데코레이터를 통해 캐싱한다.
    :param requireOnDisk_bool: 디스크에 없는 데이터는 걸러낸다.
    :return:
    '''
    mhd_list = glob.glob('F:\\gongbu/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}
    '''
    annotations.csv 파일에서 직경(diameter) 데이터를 seriesuid를 key값으로 candidate.csv 내용과 합쳐준다. 
    '''
    diameter_dict = {}
    with open('F:\\gongbu/annotations.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            diameter_dict .setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )
    '''
    candidates.csv의 데이터를 이용해서 전체 candidates 리스트를 만든다. 
    '''
    candidateInfo_list = []
    with open('F:\\gongbu/candidates.csv', 'r') as f:
        for row in list(csv.reader(f)) [1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool: # if series_uid가 없으면 디스크에 없다는 뜻이므로 스킵한다.
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            candidateDiameter_mm = 0.0
            for annotaion_tup in diameter_dict.get(series_uid, []):
                # series_uid가 없으면 [] 리턴
                annotationCenter_xyz, annotationDiameter_mm = annotaion_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:
                        # 반지름을 얻기 위해 직경을 2로 나누고 두 개의 결절 센터가 결절의 크기 기준으로 너무 떨어져 있는지의
                        # threshold를 반지름의 절반 길이로 설정한다. (실거리가 아닌 바운딩 박스로 체크)
                        break
                    else:
                        candidateDiameter_mm = annotationDiameter_mm
                        break

                candidateInfo_list.append(CandidateInforTuple(
                    isNodule_bool,
                    candidateDiameter_mm,
                    series_uid,
                    candidateCenter_xyz,
                ))
    '''
    위 코드를 정리하면 
    series_uid 컬럼을 key 값으로 삼아서 candidate와 annotation의 정보를 합친다. 
    이 때 두 행의 좌표가 충분히 가까우면 같은 결절이라고 본다.
    만약 매칭이 되지 않으면 해당 결절을 직경을 0.0이라고 본다.
    이 코드는 단순히 train 셋과 valid 셋을 나누기 위한 코드이기 때문에 결절 크기가 틀리는 것은 큰 문제가 아니다.
    다만 우리의 가정이 틀렸음은 기억해야 한다. 
    '''
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

class Ct:
    def __init__(self, series_uid):
        # 주어진 series_ui가 어떤 서브셋에 있는지 상관없으므로 와일드 카드 사용
        mhd_path = glob.glob('F:\\gongbu/subset*/{}.mhd'.format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd))

        '''
        '''
        ct_a.clip(-1000, 1000, ct_a)
        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr(
                [self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc

class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 ):
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        log.info("{!r}: {} {} samples".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
        ))

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        candidateInfo_tup = self.candidateInfo_list[ndx]
        width_irc = (32, 48, 48)

        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([
            not candidateInfo_tup.isNodule_bool,
            candidateInfo_tup.isNodule_bool
        ],
            dtype=torch.long,
        )

        return (
            candidate_t,
            pos_t,
            candidateInfo_tup.series_uid,
            torch.tensor(center_irc),
        )
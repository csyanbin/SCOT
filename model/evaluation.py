"""Provides memory buffer and logger for evaluation"""

import logging

from skimage import draw
import numpy as np
import torch


class Evaluator:
    r"""To evaluate and log evaluation metrics: PCK, LT-ACC, IoU"""
    def __init__(self, benchmark, device):
        r"""Constructor for Evaluator"""
        self.eval_buf = {
            'pfwillow': {'pck': [], 'cls_pck': dict()},
            'pfpascal': {'pck': [], 'cls_pck': dict()},
            'spair':    {'pck': [], 'cls_pck': dict()}
        }

        self.eval_funct = {
            'pfwillow': self.eval_pck,
            'pfpascal': self.eval_pck,
            'spair': self.eval_pck
        }

        self.log_funct = {
            'pfwillow': self.log_pck,
            'pfpascal': self.log_pck,
            'spair': self.log_pck
        }

        self.eval_buf = self.eval_buf[benchmark]
        self.eval_funct = self.eval_funct[benchmark]
        self.log_funct = self.log_funct[benchmark]
        self.benchmark = benchmark
        self.device = device

    def evaluate(self, prd_kps, data):
        r"""Compute desired evaluation metric"""
        return self.eval_funct(prd_kps, data)

    def log_result(self, idx, data, average=False):
        r"""Print results: PCK, or LT-ACC & IoU """
        return self.log_funct(idx, data, average)

    def eval_pck(self, prd_kps, data):
        r"""Compute percentage of correct key-points (PCK) based on prediction"""
        pckthres = data['pckthres'][0] * data['trg_intratio']
        ncorrt = correct_kps(data['trg_kps'].cuda(), prd_kps, pckthres, data['alpha'])
        pair_pck = int(ncorrt) / int(data['trg_kps'].size(1))

        self.eval_buf['pck'].append(pair_pck)

        if self.eval_buf['cls_pck'].get(data['pair_class'][0]) is None:
            self.eval_buf['cls_pck'][data['pair_class'][0]] = []
        self.eval_buf['cls_pck'][data['pair_class'][0]].append(pair_pck)

        return pair_pck

    def log_pck(self, idx, data, average):
        r"""Log percentage of correct key-points (PCK)"""
        if average:
            pck = sum(self.eval_buf['pck']) / len(self.eval_buf['pck'])
            for cls in self.eval_buf['cls_pck']:
                cls_avg = sum(self.eval_buf['cls_pck'][cls]) / len(self.eval_buf['cls_pck'][cls])
                logging.info('%15s: %3.3f' % (cls, cls_avg))
            logging.info(' * Average: %3.3f' % pck)

            return pck

        logging.info('[%5d/%5d]: \t [Pair PCK: %3.3f]\t[Average: %3.3f] %s' %
                     (idx + 1,
                      data['datalen'],
                      self.eval_buf['pck'][idx],
                      sum(self.eval_buf['pck']) / len(self.eval_buf['pck']),
                      data['pair_class'][0]))
        return None

def correct_kps(trg_kps, prd_kps, pckthres, alpha=0.1):
    r"""Compute the number of correctly transferred key-points"""
    l2dist = torch.pow(torch.sum(torch.pow(trg_kps - prd_kps, 2), 0), 0.5)
    thres = pckthres.expand_as(l2dist).float()
    correct_pts = torch.le(l2dist, thres * alpha)

    return torch.sum(correct_pts)



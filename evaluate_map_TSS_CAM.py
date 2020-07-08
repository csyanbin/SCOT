r"""Runs Semantic Correspondence as an Optimal Transport Problem"""

import argparse
import datetime
import os
import logging
import time

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

from model import scot_CAM, geometry, evaluation, util
from data.tss_dataset import TSSDataset
from data.flow import write_flo_file

import numpy as np


def create_file_path(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def run(datapath, benchmark, backbone, thres, alpha, hyperpixel,
        logpath, args, model=None, dataloader=None):
    r"""Runs Semantic Correspondence as an Optimal Transport Problem"""

    # 1. Logging initialization
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    logfile = 'logs/{}_{}_{}_{}_exp{}-{}_e{}_m{}_{}_{}'.format(benchmark,backbone,args.split,args.sim,args.exp1,args.exp2,args.eps,args.classmap,args.cam,args.hyperpixel)
    print(logfile)
    util.init_logger(logfile)
    util.log_args(args)

    # 2. Evaluation benchmark initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dataloader is None:
        csv_file = 'test_pairs_tss.csv'
        dataset = TSSDataset(csv_file=os.path.join(datapath, benchmark, csv_file),
                      dataset_path=os.path.join(datapath, benchmark)
                    )
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # 3. Model initialization
    if model is None:
        model = scot_CAM.SCOT_CAM(backbone, hyperpixel, benchmark, device, args.cam)
    else:
        model.hyperpixel_ids = util.parse_hyperpixel(hyperpixel)


    time_list = []
    for idx, data in enumerate(dataloader):
        threshold = 0.0
        
        # a) Retrieve images and adjust their sizes to avoid large numbers of hyperpixels
        if args.classmap in [0,1]:
            src_img, src_size, src_size2, src_ratio = util.resize_TSS(data['src_img'])
            trg_img, trg_size, trg_size2, trg_ratio = util.resize_TSS(data['trg_img'])

            src_mask, trg_mask, src_bbox, trg_bbox = None, None, None, None

        data['alpha'] = alpha
        #tic = time.time()

        # b) Feed a pair of images to Hyperpixel Flow model
        with torch.no_grad():
            # target meshgrids --> source matching points
            confidence_ts, trg_box, src_box = model(trg_img, src_img, args.sim, args.exp1, args.exp2, args.eps, args.classmap, trg_bbox, src_bbox, trg_mask, src_mask, backbone)

        # c) Image Grids and write flow to files
        h_tgt = int(trg_size[1].data.cpu().numpy())
        w_tgt = int(trg_size[2].data.cpu().numpy())
        grid_x_np,grid_y_np = np.meshgrid(range(1,w_tgt+1),range(1,h_tgt+1))
        grid_x = torch.tensor(grid_x_np).view(1,-1).cuda()
        grid_y = torch.tensor(grid_y_np).view(1,-1).cuda()
        trg_kps = torch.cat((grid_x,grid_y),0).type(torch.cuda.FloatTensor) # 2xwh
        trg_kps *= trg_ratio
        n_points = trg_kps.size(1)
        n_itr = int(n_points/10000)
        
        prd_kps = torch.zeros_like(trg_kps).to(trg_kps.device)
        for i in range(0, n_itr+1):
            s = i*10000
            t = min(n_points,(i+1)*10000)
            if s>=t:
                break
            trg_part = trg_kps[:, s:t].contiguous().clone()
            prd_part = geometry.predict_kps(trg_box, src_box, trg_part, confidence_ts)
            prd_kps[:,s:t] = prd_part

        def pointsToGrid (x,h_tgt=h_tgt,w_tgt=w_tgt): return x.contiguous().view(1,2,h_tgt,w_tgt).transpose(1,2).transpose(2,3)
        prd_grid = pointsToGrid(prd_kps).squeeze(0) # hxwx2 
        prd_grid /= src_ratio
        disp_x = prd_grid[:,:,0].data.cpu().numpy() - grid_x_np
        disp_y = prd_grid[:,:,1].data.cpu().numpy() - grid_y_np
        flow = np.concatenate((np.expand_dims(disp_x,2),np.expand_dims(disp_y,2)),2)

        flow_path = os.path.join(args.datapath, 'TSS/result_CAM', data['flow_path'][0])
        create_file_path(flow_path)
        write_flo_file(flow, flow_path)
        
        #toc = time.time()
        #print(idx, toc-tic)
        print(idx+1, '/', dataset.__len__())

    

if __name__ == '__main__':

    # Argument parsing
    # download TSS dataset&eval_tool manually from https://taniai.space/projects/cvpr16_dccs/
    # Put dataset into $datapath/TSS/
    parser = argparse.ArgumentParser(description='SCOT in pytorch')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--datapath', type=str, default='./Datasets_SCOT')
    parser.add_argument('--dataset', type=str, default='TSS')
    parser.add_argument('--backbone', type=str, default='resnet101')
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--hyperpixel', type=str, default='')
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--split', type=str, default='test', help='trn,val.test')
    
    # Algorithm parameters
    parser.add_argument('--sim', type=str, default='OTGeo', help='Similarity type: OT, OTGeo, cos, cosGeo')
    parser.add_argument('--exp1', type=float, default=1.0, help='exponential factor on initial cosine cost')
    parser.add_argument('--exp2', type=float, default=1.0, help='exponential factor on final OT scores')
    parser.add_argument('--eps', type=float, default=0.05, help='epsilon for Sinkhorn Regularization')
    parser.add_argument('--classmap', type=int, default=1, help='class activation map: 0 for none, 1 for using CAM')
    parser.add_argument('--cam', type=str, default='', help='extract activation map end2end rather than pre-calculation')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    run(datapath=args.datapath, benchmark=args.dataset, backbone=args.backbone, thres=args.thres,
        alpha=args.alpha, hyperpixel=args.hyperpixel, logpath=args.logpath, args=args)

    util.log_args(args)

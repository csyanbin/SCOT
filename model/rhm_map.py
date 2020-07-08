"""Implementation of optimal transport+geometric post-processing (Hough voting)"""

import math

import torch.nn.functional as F
import torch

from . import geometry


def perform_sinkhorn(C,epsilon,mu,nu,a=[],warm=False,niter=1,tol=10e-9):
    """Main Sinkhorn Algorithm"""
    if not warm:
        a = torch.ones((C.shape[0],1)) / C.shape[0]
        a = a.cuda()

    K = torch.exp(-C/epsilon)

    Err = torch.zeros((niter,2)).cuda()
    for i in range(niter):
        b = nu/torch.mm(K.t(), a)
        if i%2==0:
            Err[i,0] = torch.norm(a*(torch.mm(K, b)) - mu, p=1)
            if i>0 and (Err[i,0]) < tol:
                break

        a = mu / torch.mm(K, b)

        if i%2==0:
            Err[i,1] = torch.norm(b*(torch.mm(K.t(), a)) - nu, p=1)
            if i>0 and (Err[i,1]) < tol:
                break

        PI = torch.mm(torch.mm(torch.diag(a[:,-1]),K), torch.diag(b[:,-1]))

    del a; del b; del K
    return PI,mu,nu,Err


def appearance_similarity(src_feats, trg_feats, exp1=3):
    r"""Semantic appearance similarity (exponentiated cosine)"""
    src_feat_norms = torch.norm(src_feats, p=2, dim=1).unsqueeze(1)
    trg_feat_norms = torch.norm(trg_feats, p=2, dim=1).unsqueeze(0)
    sim = torch.matmul(src_feats, trg_feats.t()) / \
          torch.matmul(src_feat_norms, trg_feat_norms)
    sim = torch.pow(torch.clamp(sim, min=0), exp1)

    return sim


def appearance_similarityOT(src_feats, trg_feats, exp1=1.0, exp2=1.0, eps=0.05, src_weights=None, trg_weights=None):
    r"""Semantic Appearance Similarity"""
    #st_weights = src_weights.mm(trg_weights.t())

    src_feat_norms = torch.norm(src_feats, p=2, dim=1).unsqueeze(1)
    trg_feat_norms = torch.norm(trg_feats, p=2, dim=1).unsqueeze(0)
    sim = torch.matmul(src_feats, trg_feats.t()) / \
          torch.matmul(src_feat_norms, trg_feat_norms)
    sim = torch.pow(torch.clamp(sim, min=0), 1.0)
    #sim = sim*st_weights
    cost = 1-sim

    n1 = len(src_feats)
    mu = (torch.ones((n1,))/n1).cuda()
    mu = src_weights / src_weights.sum()
    n2 = len(trg_feats)
    nu = (torch.ones((n2,))/n2).cuda()
    nu = trg_weights / trg_weights.sum()
    ## ---- <Run Optimal Transport Algorithm> ----
    #mu = mu.unsqueeze(1)
    #nu = nu.unsqueeze(1)
    with torch.no_grad():
        epsilon = eps
        cnt = 0
        while True:
            PI,a,b,err = perform_sinkhorn(cost, epsilon, mu, nu)
            #PI = sinkhorn_stabilized(mu, nu, cost, reg=epsilon, numItermax=50, method='sinkhorn_stabilized', cuda=True)
            if not torch.isnan(PI).any():
                if cnt>0:
                    print(cnt)
                break
            else: # Nan encountered caused by overflow issue is sinkhorn
                epsilon *= 2.0
                #print(epsilon)
                cnt += 1

    PI = n1*PI # re-scale PI 
    #exp2 = 1.0 for spair-71k, TSS
    #exp2 = 0.5 # for pf-pascal and pfwillow
    PI = torch.pow(torch.clamp(PI, min=0), exp2)

    return PI



def hspace_bin_ids(src_imsize, src_box, trg_box, hs_cellsize, nbins_x):
    r"""Compute Hough space bin id for the subsequent voting procedure"""
    src_ptref = torch.tensor(src_imsize, dtype=torch.float).to(src_box.device)
    src_trans = geometry.center(src_box)
    trg_trans = geometry.center(trg_box)
    xy_vote = (src_ptref.unsqueeze(0).expand_as(src_trans) - src_trans).unsqueeze(2).\
                  repeat(1, 1, len(trg_box)) + \
              trg_trans.t().unsqueeze(0).repeat(len(src_box), 1, 1)

    bin_ids = (xy_vote / hs_cellsize).long()

    return bin_ids[:, 0, :] + bin_ids[:, 1, :] * nbins_x


def build_hspace(src_imsize, trg_imsize, ncells):
    r"""Build Hough space where voting is done"""
    hs_width = src_imsize[0] + trg_imsize[0]
    hs_height = src_imsize[1] + trg_imsize[1]
    hs_cellsize = math.sqrt((hs_width * hs_height) / ncells)
    nbins_x = int(hs_width / hs_cellsize) + 1
    nbins_y = int(hs_height / hs_cellsize) + 1

    return nbins_x, nbins_y, hs_cellsize


def rhm(src_hyperpixels, trg_hyperpixels, hsfilter, sim, exp1, exp2, eps, ncells=8192):
    r"""Regularized Hough matching"""
    # Unpack hyperpixels
    src_hpgeomt, src_hpfeats, src_imsize, src_weights = src_hyperpixels
    trg_hpgeomt, trg_hpfeats, trg_imsize, trg_weights = trg_hyperpixels

    # Prepare for the voting procedure
    if sim in ['cos', 'cosGeo']:
        votes = appearance_similarity(src_hpfeats, trg_hpfeats, exp1)
    if sim in ['OT', 'OTGeo']:
        votes = appearance_similarityOT(src_hpfeats, trg_hpfeats, exp1, exp2, eps, src_weights, trg_weights)
    if sim in ['OT', 'cos', 'cos2']:
        return votes

    nbins_x, nbins_y, hs_cellsize = build_hspace(src_imsize, trg_imsize, ncells)
    bin_ids = hspace_bin_ids(src_imsize, src_hpgeomt, trg_hpgeomt, hs_cellsize, nbins_x)
    hspace = src_hpgeomt.new_zeros((len(votes), nbins_y * nbins_x))

    # Proceed voting
    hbin_ids = bin_ids.add(torch.arange(0, len(votes)).to(src_hpgeomt.device).
                           mul(hspace.size(1)).unsqueeze(1).expand_as(bin_ids))
    hspace = hspace.view(-1).index_add(0, hbin_ids.view(-1), votes.view(-1)).view_as(hspace)
    hspace = torch.sum(hspace, dim=0)

    # Aggregate the voting results
    hspace = F.conv2d(hspace.view(1, 1, nbins_y, nbins_x),
                      hsfilter.unsqueeze(0).unsqueeze(0), padding=3).view(-1)

    return votes * torch.index_select(hspace, dim=0, index=bin_ids.view(-1)).view_as(votes)



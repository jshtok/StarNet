import cv2 as cv
import os
import sys

import torch
import torchvision as tv
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# from qpth.qp import QPFunction
import numpy as np
import math
from aux_routines.data_structures import assert_folder # cannot import from utils because someone imports there the ClassificationHead...
from matplotlib import cm
from aux_routines.data_structures import  get_IoU

def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.
    
    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """
    
    assert(A.dim() == 3)
    assert(B.dim() == 3)
    assert(A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1,2))


def binv(b_mat):
    """
    Computes an inverse of each matrix in the batch.
    Pytorch 0.4.1 does not support batched matrix inverse.
    Hence, we are solving AX=I.
    
    Parameters:
      b_mat:  a (n_batch, n, n) Tensor.
    Returns: a (n_batch, n, n) Tensor.
    """

    id_matrix = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat).cuda()
    b_inv, _ = torch.gesv(id_matrix, b_mat)
    
    return b_inv


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape([matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))

def MetaOptNetHead_Ridge(query, support, support_labels, n_way, n_shot, lambda_reg=50.0, double_precision=False):
    """
    Fits the support set with ridge regression and 
    returns the classification score on the query set.

    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      lambda_reg: a scalar. Represents the strength of L2 regularization.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot

    #Here we solve the dual problem:
    #Note that the classes are indexed by m & samples are indexed by i.
    #min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i

    #where w_m(\alpha) = \sum_i \alpha^m_i x_i,
    
    #\alpha is an (n_support, n_way) matrix
    kernel_matrix = computeGramMatrix(support, support)
    kernel_matrix += lambda_reg * torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

    block_kernel_matrix = kernel_matrix.repeat(n_way, 1, 1) #(n_way * tasks_per_batch, n_support, n_support)
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way) # (tasks_per_batch * n_support, n_way)
    support_labels_one_hot = support_labels_one_hot.transpose(0, 1) # (n_way, tasks_per_batch * n_support)
    support_labels_one_hot = support_labels_one_hot.reshape(n_way * tasks_per_batch, n_support)     # (n_way*tasks_per_batch, n_support)
    
    G = block_kernel_matrix
    e = -2.0 * support_labels_one_hot
    
    #This is a fake inequlity constraint as qpth does not support QP without an inequality constraint.
    id_matrix_1 = torch.zeros(tasks_per_batch*n_way, n_support, n_support)
    C = Variable(id_matrix_1)
    h = Variable(torch.zeros((tasks_per_batch*n_way, n_support)))
    dummy = Variable(torch.Tensor()).cuda()      # We want to ignore the equality constraint.

    if double_precision:
        G, e, C, h = [x.double().cuda() for x in [G, e, C, h]]

    else:
        G, e, C, h = [x.float().cuda() for x in [G, e, C, h]]

    # Solve the following QP to fit SVM:
    #        \hat z =   argmin_z 1/2 z^T G z + e^T z
    #                 subject to Cz <= h
    # We use detach() to prevent backpropagation to fixed variables.
    qp_sol = QPFunction(verbose=False)(G, e.detach(), C.detach(), h.detach(), dummy.detach(), dummy.detach())
    #qp_sol = QPFunction(verbose=False)(G, e.detach(), dummy.detach(), dummy.detach(), dummy.detach(), dummy.detach())

    #qp_sol (n_way*tasks_per_batch, n_support)
    qp_sol = qp_sol.reshape(n_way, tasks_per_batch, n_support)
    #qp_sol (n_way, tasks_per_batch, n_support)
    qp_sol = qp_sol.permute(1, 2, 0)
    #qp_sol (tasks_per_batch, n_support, n_way)
    
    # Compute the classification score.
    compatibility = computeGramMatrix(support, query)
    compatibility = compatibility.float()
    compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
    qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
    logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
    logits = logits * compatibility
    logits = torch.sum(logits, 1)

    return logits

def R2D2Head(query, support, support_labels, n_way, n_shot, l2_regularizer_lambda=50.0):
    """
    Fits the support set with ridge regression and 
    returns the classification score on the query set.
    
    This model is the classification head described in:
    Meta-learning with differentiable closed-form solvers
    (Bertinetto et al., in submission to NIPS 2018).
    
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      l2_regularizer_lambda: a scalar. Represents the strength of L2 regularization.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

    id_matrix = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()
    
    # Compute the dual form solution of the ridge regression.
    # W = X^T(X X^T - lambda * I)^(-1) Y
    ridge_sol = computeGramMatrix(support, support) + l2_regularizer_lambda * id_matrix
    ridge_sol = binv(ridge_sol)
    ridge_sol = torch.bmm(support.transpose(1,2), ridge_sol)
    ridge_sol = torch.bmm(ridge_sol, support_labels_one_hot)
    
    # Compute the classification score.
    # score = W X
    logits = torch.bmm(query, ridge_sol)

    return logits


def MetaOptNetHead_SVM_He(query, support, support_labels, n_way, n_shot, C_reg=0.01, double_precision=False):
    """
    Fits the support set with multi-class SVM and 
    returns the classification score on the query set.
    
    This is the multi-class SVM presented in:
    A simplified multi-class support vector machine with reduced dual optimization
    (He et al., Pattern Recognition Letter 2012).
    
    This SVM is desirable because the dual variable of size is n_support
    (as opposed to n_way*n_support in the Weston&Watkins or Crammer&Singer multi-class SVM).
    This model is the classification head that we have initially used for our project.
    This was dropped since it turned out that it performs suboptimally on the meta-learning scenarios.
    
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      C_reg: a scalar. Represents the cost parameter C in SVM.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot

    
    kernel_matrix = computeGramMatrix(support, support)

    V = (support_labels * n_way - torch.ones(tasks_per_batch, n_support, n_way).cuda()) / (n_way - 1)
    G = computeGramMatrix(V, V).detach()
    G = kernel_matrix * G
    
    e = Variable(-1.0 * torch.ones(tasks_per_batch, n_support))
    id_matrix = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support)
    C = Variable(torch.cat((id_matrix, -id_matrix), 1))
    h = Variable(torch.cat((C_reg * torch.ones(tasks_per_batch, n_support), torch.zeros(tasks_per_batch, n_support)), 1))
    dummy = Variable(torch.Tensor()).cuda()      # We want to ignore the equality constraint.

    if double_precision:
        G, e, C, h = [x.double().cuda() for x in [G, e, C, h]]
    else:
        G, e, C, h = [x.cuda() for x in [G, e, C, h]]
        
    # Solve the following QP to fit SVM:
    #        \hat z =   argmin_z 1/2 z^T G z + e^T z
    #                 subject to Cz <= h
    # We use detach() to prevent backpropagation to fixed variables.
    qp_sol = QPFunction(verbose=False)(G, e.detach(), C.detach(), h.detach(), dummy.detach(), dummy.detach())

    # Compute the classification score.
    compatibility = computeGramMatrix(query, support)
    compatibility = compatibility.float()

    logits = qp_sol.float().unsqueeze(1).expand(tasks_per_batch, n_query, n_support)
    logits = logits * compatibility
    logits = logits.view(tasks_per_batch, n_query, n_shot, n_way)
    logits = torch.sum(logits, 2)

    return logits

is_starnet_init = False
vyx, mxx, mxy, targX, targY = None, None, None, None, None
kernel, kernel_size = None, None
g_b, g_nQ, g_nC, g_nS = 0, 0, 0, 0

def starNet_init(b, nQ, nC, nS, y, x, voting_kernel_sigma=1):
    global is_starnet_init
    global vyx, mxx, mxy, targX, targY
    global kernel, kernel_size
    global g_b, g_nQ, g_nC, g_nS

    if (not is_starnet_init) or ((g_b, g_nQ, g_nC, g_nS) != (b, nQ, nC, nS)):
        # gx, gy = torch.meshgrid(torch.arange(0,x,dtype=torch.float),torch.arange(0,y,dtype=torch.float))
        gy, gx = torch.meshgrid(torch.arange(0, y, dtype=torch.float), torch.arange(0, x, dtype=torch.float))

        gx = gx.contiguous().view(-1, 1)
        gy = gy.contiguous().view(-1, 1)
        ox = (x / 2.0 - gx).view(1, -1)
        oy = (y / 2.0 - gy).view(1, -1)
        vx = gx + ox
        vy = gy + oy
        mnx = vx.min()
        mny = vy.min()
        vx -= mnx
        vy -= mny
        vx, vy = vx.type(torch.LongTensor).view(-1), vy.type(torch.LongTensor).view(-1)
        mxx = vx.max().item() + 1
        mxy = vy.max().item() + 1
        vyx = vx + vy * mxx  # assume the order is y first x second
        vyx = vyx.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(b, nQ, nS, -1).cuda()

        # prepare back projection targets
        targ = vyx.view(b * nQ * nS, x * y, x * y)  # b*nQ*nC, x * y , x * y
        targX = targ.fmod(mxx)
        targY = (targ - targX) // mxx

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel_size = [5] * 2
        sigma = [ voting_kernel_sigma ] * 2
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        kernel = kernel.unsqueeze(0).unsqueeze(0) / kernel.max()
        kernel = kernel.cuda()

        g_b, g_nQ, g_nC, g_nS = b, nQ, nC, nS
        is_starnet_init = True

def comp_proximity_map(        support, query,
        n_way, normalize, vote_prob_nrm_type, multiple_hypotheses_perc,
        do_back_project = True, m_sup=None, m_query=None, voting_kernel_sigma=1, bp_sigma=2.0
):
    global is_starnet_init
    global vyx, mxx, mxy, targX, targY
    global kernel, kernel_size
    global g_b, g_nQ, g_nC

    bp_query_, bp_support = None, None

    b, nQ, ch, y, x = query.shape
    nS = support.shape[1]
    nC = n_way

    starNet_init(b, nQ, nC, nS, y, x, voting_kernel_sigma)

    if normalize:
        query_n = F.normalize(query, p=2, dim=2)
        support_n = F.normalize(support, p=2, dim=2)
    else:
        query_n, support_n = query, support

    query_ = query_n.permute(0, 1, 3, 4, 2).contiguous().view(b, -1, ch)  # b, nQ * y * x, ch
    support_ = support_n.permute(0, 2, 1, 3, 4).contiguous().view(b, ch, -1)  # b, ch, nS * y * x
    qs = torch.sum(torch.pow(query_, 2), dim=2, keepdim=True)  # b, nQ * y * x, 1
    ss = torch.sum(torch.pow(support_, 2), dim=1, keepdim=True)  # b, 1, nS * y * x
    d = qs + ss - 2.0 * torch.matmul(query_, support_)  # b, nQ * y * x, nS * y * x
    if not normalize:
        d /= float(ch)
    sigma = 0.2
    p = torch.exp(-0.5 * d / (sigma ** 2))  # b, nQ * y * x, nS * y * x
    p = p.view(b, nQ, y * x, nS, y * x).permute(0, 1, 3, 2,
                                                4)  # b, nQ, nS, y * x, y * x (source is query, target is support)
    return p

def votingFun(
        support, query,
        n_way, normalize, vote_prob_nrm_type, multiple_hypotheses_perc,
        do_back_project = True, m_sup=None, m_query=None, voting_kernel_sigma=1, bp_sigma=2.0
):
    global is_starnet_init
    global vyx, mxx, mxy, targX, targY
    global kernel, kernel_size
    global g_b, g_nQ, g_nC

    bp_query_, bp_support = None, None

    b, nQ, ch, y, x = query.shape
    nS = support.shape[1]
    nC = n_way

    starNet_init(b, nQ, nC, nS, y, x, voting_kernel_sigma)

    if normalize:
        query_n = F.normalize(query, p=2, dim=2)
        support_n = F.normalize(support, p=2, dim=2)
    else:
        query_n, support_n = query, support

    query_ = query_n.permute(0, 1, 3, 4, 2).contiguous().view(b, -1, ch)  # b, nQ * y * x, ch
    support_ = support_n.permute(0, 2, 1, 3, 4).contiguous().view(b, ch, -1)  # b, ch, nS * y * x
    qs = torch.sum(torch.pow(query_, 2), dim=2, keepdim=True)  # b, nQ * y * x, 1
    ss = torch.sum(torch.pow(support_, 2), dim=1, keepdim=True)  # b, 1, nS * y * x
    d = qs + ss - 2.0 * torch.matmul(query_, support_)  # b, nQ * y * x, nS * y * x
    if not normalize:
        d /= float(ch)
    sigma = 0.2
    p = torch.exp(-0.5 * d / (sigma ** 2))  # b, nQ * y * x, nS * y * x
    p = p.view(b, nQ, y * x, nS, y * x).permute(0, 1, 3, 2,
                                                4)  # b, nQ, nS, y * x, y * x (source is query, target is support)

    # normalize to make the votes conditional probabilities on all support and all support locations
    vnrm = None
    if vote_prob_nrm_type == 1:
        vnrm = (p.sum(dim=(-1, -3), keepdim=True) + 1e-20)
    elif vote_prob_nrm_type == 2:
        vnrm = (p.max(dim=-1, keepdim=True).values.max(dim=-3, keepdim=True).values + 1e-20)
    elif vote_prob_nrm_type == 3:
        vnrm = (p.max(dim=-1, keepdim=True).values + 1e-20)
    elif vote_prob_nrm_type == 4:
        vnrm = (p.sum(dim=-1, keepdim=True) + 1e-20)
    if vnrm is not None:
        p = p / vnrm

    accp = p.contiguous().view(b, nQ, nS, -1)  # b, nQ, nS, y * x * y * x
    # support_labels_one_hot = support_labels_one_hot.view(b, 1, nS, nC).transpose(2,3) # b, 1, nC, nS
    # accp = torch.matmul(support_labels_one_hot,p.contiguous().view(b,nQ,nS,-1)) # (b,1,nC,nS) x (b,nQ,nS,(y*x)^2)
    # # accp shape is now b, nQ, nC, y * x * y * x (source is query, target is classes)
    # # accp = accp.view(b, nQ, nC, y * x, y * x) # b, nQ, nC, y * x, y * x

    if (m_sup is not None) or (m_query is not None):
        accp = accp.view(b, nQ, nS, y * x, y * x)

    if m_sup is not None:
        # rows are query, columns are support
        accp *= m_sup.view(b, nQ, nS, 1, y * x)

    if m_query is not None:
        # rows are query, columns are support
        accp *= m_query.view(b, nQ, nS, y * x, 1)

    if (m_sup is not None) or (m_query is not None):
        accp = accp.view(b, nQ, nS, -1)

    nSteps = 1
    for iStep in range(nSteps):
        if iStep > 0:
            accp = accp.view(b, nQ, nS, y * x, y * x)

            mask = bp_query_.view(b * nQ, nS, y * x, 1)  # .max(dim=2, keepdim=True).values #TODO: fix, this max is wrong
            scores = torch.max(vm_.view(b, nQ, nS, -1), dim=3).values  # b, nQ, nS
            c = torch.max(scores, dim=2).indices  # b, nQ
            mask = torch.cat([mask[j, i].unsqueeze(0) for j, i in zip(range(b * nQ), c.view(-1))], dim=0)
            mask = mask.view(b, nQ, 1, y * x, 1)
            mask /= mask.sum(dim=-2, keepdim=True)

            # mask = bp_support.mean(dim=1, keepdim=True).view(b, 1, nS, 1, y * x)
            # # mask /= mask.max(dim=-1, keepdim=True).values
            # # mask /= mask.sum(dim=-1, keepdim=True)

            accp *= mask
            accp = accp.view(b, nQ, nS, -1)

        vm = torch.zeros((b, nQ, nS, mxy * mxx), dtype=torch.float).cuda()
        vm.scatter_add_(dim=3, index=vyx, src=accp)
        vm = vm.view(b * nQ * nS, 1, mxy, mxx)  # b*nQ*nS, 1, mxy, mxx

        vm_ = F.conv2d(vm, kernel, bias=None, stride=1, padding=int((kernel_size[0] - 1) / 2), dilation=1, groups=1)

        if do_back_project and ((iStep < (nSteps - 1)) or (nSteps == 1)):
            bp_vm = vm_.view(b, nQ, nS, mxy, mxx)
            bp_query_, bp_support, predY, predX, refVal = back_project(bp_vm, accp, None, [y, x], kernel_size, max_only=False, bp_normalize=False,bp_sigma=bp_sigma)
            bp_query_set = []
            vm_set = []
            if multiple_hypotheses_perc < 1.0:
                bp_vm = bp_vm.clone().detach()
                while True:
                    # supress the current maxima in bp_vm
                    bp_vm = bp_vm.view(b, nQ, nS, -1)
                    # predY is b, nQ, nS
                    # predX is b, nQ, nS
                    hsz = [1, 1]  # [int((x - 1) / 2) for x in kernel_size]
                    for oy in range(-hsz[0], hsz[0] + 1):
                        for ox in range(-hsz[1], hsz[1] + 1):
                            ty = (predY + oy).clamp(0, mxy - 1)
                            tx = (predX + ox).clamp(0, mxx - 1)
                            ix = (ty * mxx + tx).unsqueeze(-1)
                            bp_vm.scatter_(dim=3, index=ix, value=0)
                    bp_vm = bp_vm.view(b, nQ, nS, mxy, mxx)
                    vm_set.append(bp_vm)
                    # now re-run back-projection to obtain the next set of maps
                    bpq_next, bps_next, predY, predX, curMaxVal = back_project(bp_vm, accp, None, [y, x], kernel_size, max_only=False, bp_normalize=False,bp_sigma=bp_sigma)
                    updFilter = (curMaxVal / (refVal + 1e-7)) >= multiple_hypotheses_perc
                    if not updFilter.any():
                        break
                    curr_bp_query = updFilter.type(torch.cuda.FloatTensor).unsqueeze(-1).unsqueeze(-1) * bpq_next
                    bp_query_set.append(curr_bp_query)
                    bp_query_ = torch.max(bp_query_, curr_bp_query)
                    bp_support = torch.max(bp_support, updFilter.type(torch.cuda.FloatTensor).unsqueeze(-1).unsqueeze(-1) * bps_next)

    return vm_, bp_query_, bp_support, b, nQ, nS, nC, y, x, ch,vm_set, bp_query_set

def StarNetHead(query, support, support_labels, n_way, n_shot, s2_query=None, s2_support=None, tb=None, inputs=None, tb_prefix='', opt=None,Nbatch=0,
                two_stage=True, normalize=True, allow_stage_cross_talk=False, stage2_mode=3, vote_prob_nrm_type=0, multiple_hypotheses_perc=1.0,
                stage2_cosine=False, support_comb_type='avg', stage2_type='proto',img_save_path=None):
    '''

    :param query: embeddings of the query samples (emb_query = embedding_net_st1(...)
    :param support:
    :param support_labels:      labels_support
    :param n_way:
    :param n_shot:
    :param s2_query:
    :param s2_support:
    :param tb:
    :param inputs:              inputs=(data_support, data_query)
    :param tb_prefix:
    :param opt:
    :param two_stage:
    :param normalize:
    :param allow_stage_cross_talk:
    :param stage2_mode:
    :param vote_prob_nrm_type:
    :param multiple_hypotheses_perc:
    :param stage2_cosine:
    :param support_comb_type:
    :param stage2_type:
    :return:
    '''
    global is_starnet_init
    global vyx, mxx, mxy, targX, targY
    global kernel, kernel_size
    global g_b, g_nQ, g_nC

    #img_save_path = '.'
    mean_clr = [120.39586422, 115.59361427, 104.54012653]
    std_clr = [70.68188272, 68.27635443, 72.54505529]
    voting_kernel_sigma = 1
    bp_sigma = 2.0
    if opt is not None:
        if 'two_stage' in opt:
            two_stage = bool(opt.two_stage)
        if 'normalize' in opt:
            normalize = bool(opt.normalize)
        if 'allow_stage_cross_talk' in opt:
            allow_stage_cross_talk = bool(opt.allow_stage_cross_talk)
        if 'stage2_mode' in opt:
            stage2_mode = opt.stage2_mode # was bool(opt.stage2_mode)
        if 'stage2_type' in opt:
            stage2_type = opt.stage2_type
        if 'save_path' in opt and img_save_path is None:
            img_save_path = opt.save_path
        if 'load' in opt and img_save_path is None:
            img_save_path = assert_folder(os.path.join(os.path.split(opt.load)[0],'dets_{0}'.format(Nbatch)))
        if 'mean_clr' in opt:
            mean_clr = opt.mean_clr
        if 'std_clr' in opt:
            std_clr = opt.std_clr
        if 'vote_prob_nrm_type' in opt:
            vote_prob_nrm_type = opt.vote_prob_nrm_type
        if 'multiple_hypotheses_perc' in opt:
            multiple_hypotheses_perc = opt.multiple_hypotheses_perc
        if 'stage2_cosine' in opt:
            stage2_cosine = opt.stage2_cosine
        if 'support_comb_type' in opt:
            support_comb_type = opt.support_comb_type
        if 'voting_kernel_sigma' in opt:
            voting_kernel_sigma = opt.voting_kernel_sigma
        if 'bp_sigma' in opt:
            bp_sigma = opt.bp_sigma

    vm_, bp_query_, bp_support, b, nQ, nS, nC, y, x, ch,vm_set, bp_query_set = votingFun(
        support, query,
        n_way, normalize, vote_prob_nrm_type, multiple_hypotheses_perc,
        do_back_project=True, m_sup=None, m_query=None, voting_kernel_sigma=voting_kernel_sigma, bp_sigma=bp_sigma
    )

    bp_query, vm, support_labels_one_hot = support2class(bp_query_, vm_, support_labels, b, nQ, nS, nC, y, x, comb_type=support_comb_type)


    #normalize the individual bp maps
    bp_query_ /= (bp_query_.max(dim=-2, keepdim=True).values.max(dim=-1, keepdim=True).values + 1e-7)

    scores = torch.max(vm, dim=3).values  # b, nQ, nC
    c = torch.max(scores, dim=2).indices  # b, nQ
    # ---------------------------------------------------
    prox_query_set = [[] for _ in range(nQ)]
    if True:
        for q in range(nQ):
            query_q = query[:, q].unsqueeze(1)
            prox_query = comp_proximity_map(
                query_q, query_q,
                n_way, normalize, vote_prob_nrm_type, multiple_hypotheses_perc,
                do_back_project=True, m_sup=None, m_query=None, voting_kernel_sigma=voting_kernel_sigma, bp_sigma=bp_sigma
            )
            prox_query = prox_query.squeeze(0).squeeze(0).squeeze(0)
            prox_query_set[q] = prox_query

    if False: #gen individual heatmaps for each NMS iteration
        scores_loc_set = []
        c_loc_set = []
        for loc_vm, loc_bp_query in zip(vm_set, bp_query_set):
            bp_query_loc, vm_loc_, _ = support2class(loc_bp_query, vm_, support_labels, b, nQ, nS, nC, y, x,
                                                                 comb_type=support_comb_type)
            scores_loc = torch.max(vm_loc_, dim=3).values
            scores_loc_set.append(scores_loc)  # b, nQ, nC
            c_loc_set.append(torch.max(scores_loc, dim=2).indices)  # b, nQ
    else:
        scores_loc_set = None
        c_loc_set = None

    DEBUG = False
    SHOW_QUERY_BP = True
    SHOW_SUPPORT_BP = True
    SHOW_VM = False
    if DEBUG and (inputs is not None) and (tb is not None):
        bb = 0  # batch episode to show
        report_images(
            bb, inputs, mean_clr, std_clr, support_labels, c,
            bp_support, bp_query, vm, b, nQ, nC, mxy, mxx,
            img_save_path, x, y, tb=tb, tb_prefix=tb_prefix,
            SHOW_VM=SHOW_VM, SHOW_QUERY_BP=SHOW_QUERY_BP, SHOW_SUPPORT_BP=SHOW_SUPPORT_BP,
            prefix=''
        )

        report_images_paired(
                bb, inputs, mean_clr, std_clr, support_labels, c,
                bp_support, bp_query, vm, b, nQ, nC, mxy, mxx,
                img_save_path, x, y      )
    # report_images_all_supports(
    #     bb, inputs, mean_clr, std_clr, support_labels, c,
    #     bp_support, bp_query, vm, b, nQ, nC, mxy, mxx,
    #     img_save_path, x, y)

    # report_images_all_supports_all_bps(
    #     bb, inputs, mean_clr, std_clr, support_labels, c,
    #     bp_support, bp_query, vm, b, nQ, nC, mxy, mxx,
    #     img_save_path, x, y)
    #
    # report_images_matching_supports_bps(
    #     bb, inputs, mean_clr, std_clr, support_labels, c,
    #     bp_support, bp_query, vm, b, nQ, nC, mxy, mxx,
    #     img_save_path, x, y)


    if two_stage:
        if (s2_query is not None) and (s2_support is not None):
            query_src, support_src = s2_query, s2_support
        else:
            query_src, support_src = query, support

        # apply second stage classifier using attention
        m_query = bp_query_.view(b, nQ, nS, -1, 1)
        m_sup = bp_support.view(b, nQ, nS, -1, 1)

        if stage2_type == 'proto':
            # normalize the maps so the weights they provide sum to 1
            m_query = m_query / (m_query.sum(dim=-2, keepdim=True) + 1e-7)
            m_sup = m_sup / (m_sup.sum(dim=-2, keepdim=True) + 1e-7)

            if not allow_stage_cross_talk:
                m_query = m_query.detach()
                m_sup = m_sup.detach()

            stage2_scores = stage2_proto(
                support_src, query_src,
                m_sup, m_query,
                support_labels_one_hot,
                b, nQ, nS, ch,
                stage2_mode, stage2_cosine
            )
        elif stage2_type == 'star':
            # # normalize the maps so the weights they provide max to 1
            # m_query = m_query / (m_query.max(dim=-2, keepdim=True).values + 1e-7)
            # m_sup = m_sup / (m_sup.max(dim=-2, keepdim=True).values + 1e-7)
            # normalize the maps so the weights they sum to 1
            m_query = m_query / (m_query.sum(dim=-2, keepdim=True) + 1e-7)
            m_sup = m_sup / (m_sup.sum(dim=-2, keepdim=True) + 1e-7)

            if not allow_stage_cross_talk:
                m_query = m_query.detach()
                m_sup = m_sup.detach()

            do_back_project = DEBUG and (inputs is not None) and (tb is not None)

            #otherwise we get out of memory
            stage2_multiple_hypotheses_perc = 1.0

            stage2_vm_, stage2_bp_query_, stage2_bp_support, _, _, _, _, _, _, _ = votingFun(
                support_src, query_src,
                n_way, normalize, vote_prob_nrm_type, stage2_multiple_hypotheses_perc,
                do_back_project=do_back_project,
                m_sup=m_sup if (stage2_mode in [2, 3]) else None,
                m_query=m_query if (stage2_mode in [1, 3]) else None,
                bp_sigma=bp_sigma
            )
            stage2_bp_query, stage2_vm, _ = support2class(stage2_bp_query_, stage2_vm_, support_labels, b, nQ, nS, nC, y, x, comb_type=support_comb_type)
            stage2_scores = torch.max(stage2_vm, dim=3).values  # b, nQ, nC
            stage2_c = torch.max(stage2_scores, dim=2).indices  # b, nQ

            if do_back_project:
                bb = 0  # batch episode to show
                report_images(
                    bb, inputs, mean_clr, std_clr, support_labels,
                    stage2_c, stage2_bp_support, stage2_bp_query, stage2_vm,
                    b, nQ, nC, mxy, mxx,
                    img_save_path, x, y, tb=tb, tb_prefix=tb_prefix,
                    SHOW_VM=SHOW_VM, SHOW_QUERY_BP=SHOW_QUERY_BP, SHOW_SUPPORT_BP=SHOW_SUPPORT_BP,
                    prefix='stage2_'
                )
        else:
            raise Exception('unknown stage 2 type: {}'.format(stage2_type))

        scores = [scores, stage2_scores]

    return scores, bp_query, c,prox_query_set

def stage2_proto(
    support_src, query_src,
    m_sup, m_query,
    support_labels_one_hot,
    b, nQ, nS, ch,
    stage2_mode, stage2_cosine
):
    if stage2_mode in [1, 3]:
        query_p = torch.matmul(query_src.unsqueeze(2).expand(-1, -1, nS, -1, -1, -1).view(b, nQ, nS, ch, -1),
                               m_query).squeeze(-1)  # b, nQ, nS, ch
    else:
        query_p = query_src.unsqueeze(2).expand(-1, -1, nS, -1, -1, -1).view(b, nQ, nS, ch, -1).mean(
            dim=-1)  # b, nQ, nS, ch

    if stage2_mode in [2, 3]:
        sup_p = torch.matmul(support_src.unsqueeze(1).expand(-1, nQ, -1, -1, -1, -1).view(b, nQ, nS, ch, -1),
                             m_sup).squeeze(-1)  # b, nQ, nS, ch
    else:
        sup_p = support_src.unsqueeze(1).expand(-1, nQ, -1, -1, -1, -1).view(b, nQ, nS, ch, -1).mean(
            dim=-1)  # b, nQ, nS, ch

    # support_labels_one_hot = b, nS, nC
    SL = support_labels_one_hot.transpose(-1, -2).unsqueeze(1)  # b, 1, nC, nS
    sup_p = torch.matmul(SL, sup_p)  # b, nQ, nC, ch
    query_p = torch.matmul(SL, query_p)  # b, nQ, nC, ch
    if not stage2_cosine:
        D = (sup_p * sup_p + query_p * query_p - 2.0 * sup_p * query_p).sum(dim=-1)  # b, nQ, nC
        D /= float(ch)
        scores = -D
    else:
        D = (sup_p * query_p).sum(dim=-1) / (
                    torch.sqrt((sup_p * sup_p).sum(dim=-1)) * torch.sqrt((query_p * query_p).sum(dim=-1)))
        scores = D

    return scores

def report_images_paired(bb, inputs, mean_clr, std_clr, support_labels, c,bp_support, bp_query, vm, b, nQ, nC, mxy, mxx, img_save_path,x, y):
    with torch.no_grad():
        mean_pix = torch.Tensor([x / 255.0 for x in mean_clr]).unsqueeze(-1).unsqueeze(-1).cuda()
        std_pix = torch.Tensor([x / 255.0 for x in std_clr]).unsqueeze(-1).unsqueeze(-1).cuda()

        # denormalize and prepare the original query images

        orig = inputs[1][bb]
        orig = [x * std_pix + mean_pix for x in orig]


        L = support_labels[bb].cpu().numpy()
        imset = []
        for i_q in range(nQ):
            ix = np.where(L == c[bb, i_q].item())[0]
            sup_img = inputs[0][bb, ix[0]]
            sup_img = sup_img * std_pix + mean_pix
            sup_bp = bp_support[bb, i_q, ix[0]].detach().cpu().numpy()  # c[bb, i_q]
            sup_bp = torch.Tensor(cm.jet(sup_bp / sup_bp.max())[:, :, :-1]).permute(2, 0, 1)
            sup_blend = blend([sup_bp], [sup_img], normalize=False)

            query_img = orig[i_q]
            if len(bp_query.shape) == 5:
                query_bp = bp_query[bb, i_q, int(c[bb, i_q].item())].detach().cpu().numpy()
            else:
                query_bp = bp_query[bb, i_q].detach().cpu().numpy()
            query_bp = torch.Tensor(cm.jet(query_bp / query_bp.max())[:, :, :-1]).permute(2, 0, 1)
            query_blend = blend([query_bp], [query_img], normalize=False)
            hor_line = np.ones((3,8,query_blend[0].shape[1]))
            pair = torch.from_numpy(np.concatenate( (np.asarray(query_blend[0].cpu()),hor_line, np.asarray(sup_blend[0].cpu())),axis=1))
            imset.append(pair)
            tv.utils.save_image(pair, os.path.join(img_save_path, 'query_supp_{}.png'.format(i_q)))

        gridBP_sup = tv.utils.make_grid(imset, nrow=nQ, normalize=True, pad_value=0.5)
        tv.utils.save_image(gridBP_sup, os.path.join(img_save_path, 'query_supp_pairs.png'))

def report_images_all_supports(bb, inputs, mean_clr, std_clr, support_labels, c
                               ,bp_support, bp_query, vm, b, nQ, nC, mxy, mxx, img_save_path,x, y):
    with torch.no_grad():
        mean_pix = torch.Tensor([x / 255.0 for x in mean_clr]).unsqueeze(-1).unsqueeze(-1).cuda()
        std_pix = torch.Tensor([x / 255.0 for x in std_clr]).unsqueeze(-1).unsqueeze(-1).cuda()

        # denormalize and prepare the original query images

        orig = inputs[1][bb]
        orig = [x * std_pix + mean_pix for x in orig]


        L = support_labels[bb].cpu().numpy()
        imset = []
        nS = inputs[0].shape[1]
        for i_q in range(nQ):
            query_img = orig[i_q]
            if len(bp_query.shape) == 5:
                query_bp = bp_query[bb, i_q, int(c[bb, i_q].item())].detach().cpu().numpy()
            else:
                query_bp = bp_query[bb, i_q].detach().cpu().numpy()
            query_bp = torch.Tensor(cm.jet(query_bp / query_bp.max())[:, :, :-1]).permute(2, 0, 1)
            query_blend = blend([query_bp], [query_img], normalize=False)
            q_set =  np.asarray(query_blend[0].cpu())
            hor_line = np.ones((3, 5, q_set.shape[1]))
            for i_s in range(nS):
                #ix = np.where(L == c[bb, i_q].item())[0]
                sup_img = inputs[0][bb,i_s]
                sup_img = sup_img * std_pix + mean_pix
                sup_bp = bp_support[bb, i_q, i_s].detach().cpu().numpy()  # c[bb, i_q]
                sup_bp = torch.Tensor(cm.jet(sup_bp / sup_bp.max())[:, :, :-1]).permute(2, 0, 1)
                sup_blend = blend([sup_bp], [sup_img], normalize=False)
                q_set = np.concatenate((q_set, np.asarray(sup_blend[0].cpu())),axis=2)
            q_set = torch.from_numpy(q_set)
            imset.append(q_set)
            tv.utils.save_image(q_set, os.path.join(img_save_path, 'query_supp_{}.png'.format(i_q)))
        gridBP_sup = tv.utils.make_grid(imset, nrow=1, normalize=True, pad_value=0.5)
        tv.utils.save_image(gridBP_sup, os.path.join(os.path.dirname(img_save_path),'q_all_{}.jpg'.format(os.path.basename(img_save_path).split('_')[1])))

def report_images_all_supports_all_bps(bb, inputs, mean_clr, std_clr, support_labels, c
                               ,bp_support, bp_query, vm, b, nQ, nC, mxy, mxx, img_save_path,x, y):
    epi_num = os.path.basename(img_save_path).split('_')[1]
    with torch.no_grad():
        mean_pix = torch.Tensor([x / 255.0 for x in mean_clr]).unsqueeze(-1).unsqueeze(-1).cuda()
        std_pix = torch.Tensor([x / 255.0 for x in std_clr]).unsqueeze(-1).unsqueeze(-1).cuda()

        # denormalize and prepare the original query images

        orig = inputs[1][bb]
        orig = [x * std_pix + mean_pix for x in orig]

        L = support_labels[bb].cpu().numpy()

        nS = inputs[0].shape[1]
        for i_q in range(nQ):
            query_img = orig[i_q]
            bp_grid=[]
            query_imset = []
            for i_b in range(nS):
                if len(bp_query.shape) == 5:
                    query_bp = bp_query[bb, i_q, i_b].detach().cpu().numpy()
                else:
                    query_bp = bp_query[bb, i_q].detach().cpu().numpy()
                query_bp = torch.Tensor(cm.jet(query_bp / query_bp.max())[:, :, :-1]).permute(2, 0, 1)
                query_blend = blend([query_bp], [query_img], normalize=False)
                q_set =  np.asarray(query_blend[0].cpu())

                for i_s in range(nS):
                    #ix = np.where(L == c[bb, i_q].item())[0]
                    sup_img = inputs[0][bb,i_s]
                    sup_img = sup_img * std_pix + mean_pix
                    sup_bp = bp_support[bb, i_q, i_b].detach().cpu().numpy()  # c[bb, i_q]
                    sup_bp = torch.Tensor(cm.jet(sup_bp / sup_bp.max())[:, :, :-1]).permute(2, 0, 1)
                    sup_blend = blend([sup_bp], [sup_img], normalize=False)
                    q_set = np.concatenate((q_set, np.asarray(sup_blend[0].cpu())),axis=2)
                q_set = torch.from_numpy(q_set)
                query_imset.append(q_set)
            query_grid = tv.utils.make_grid(query_imset, nrow=1, normalize=True, pad_value=0.5)
            tv.utils.save_image(query_grid, os.path.join(os.path.dirname(img_save_path), 'query_set_{}_{}.jpg'.format(i_q,epi_num)))
            #tv.utils.save_image(q_set, os.path.join(img_save_path, 'query_supp_{}.png'.format(i_q)))
        #gridBP_sup = tv.utils.make_grid(imset, nrow=1, normalize=True, pad_value=0.5)
        #tv.utils.save_image(gridBP_sup, os.path.join(os.path.dirname(img_save_path),'q_all_{}.jpg'.format(epi_num)))

def report_images_matching_supports_bps(bb, inputs, mean_clr, std_clr, support_labels, c
                               ,bp_support, bp_query, vm, b, nQ, nC, mxy, mxx, img_save_path,x, y):
    epi_num = os.path.basename(img_save_path).split('_')[1]
    with torch.no_grad():
        mean_pix = torch.Tensor([x / 255.0 for x in mean_clr]).unsqueeze(-1).unsqueeze(-1).cuda()
        std_pix = torch.Tensor([x / 255.0 for x in std_clr]).unsqueeze(-1).unsqueeze(-1).cuda()

        # denormalize and prepare the original query images

        orig = inputs[1][bb]
        orig = [x * std_pix + mean_pix for x in orig]

        L = support_labels[bb].cpu().numpy()

        nS = inputs[0].shape[1]
        for i_q in range(nQ):
            query_img = orig[i_q]
            bp_grid=[]
            query_imset = []
            for i_s in range(nS):
                # ix = np.where(L == c[bb, i_q].item())[0]
                sup_img = inputs[0][bb, i_s]  # support image
                sup_img = sup_img * std_pix + mean_pix

                # bp_query index for i_q, i_s pair:
                query_bp = bp_query[bb, i_q, i_s].detach().cpu().numpy()
                # bp_supp  index for i_q, i_s pair:
                sup_bp = bp_support[bb, i_q, i_s].detach().cpu().numpy()

                # prep. query, supp blended images----------------------------
                query_bp = torch.Tensor(cm.jet(query_bp / query_bp.max())[:, :, :-1]).permute(2, 0, 1)
                query_blend = blend([query_bp], [query_img], normalize=False)

                sup_bp = torch.Tensor(cm.jet(sup_bp / sup_bp.max())[:, :, :-1]).permute(2, 0, 1)
                sup_blend = blend([sup_bp], [sup_img], normalize=False)

                # prep vertical q-s paits
                #hor_line = np.ones((3, 8, query_blend[0].shape[1]))
                #pair = torch.from_numpy(np.concatenate((np.asarray(query_blend[0]), hor_line, np.asarray(sup_blend[0])), axis=1))
                pair = torch.cat((query_blend[0],sup_blend[0]),dim=1)
                query_imset.append(pair)

            # make the array of pairs: ------------------------------------
            query_grid = tv.utils.make_grid(query_imset, nrow=nS, normalize=True, pad_value=0.5)
            tv.utils.save_image(query_grid, os.path.join(os.path.dirname(img_save_path), 'query_match_{}_{}.jpg'.format(i_q, epi_num)))

def report_images(bb, inputs, mean_clr, std_clr, support_labels, c,bp_support, bp_query, vm, b, nQ, nC, mxy, mxx, img_save_path,
                  x, y, tb=None, tb_prefix='',
                  SHOW_VM=True, SHOW_QUERY_BP=True, SHOW_SUPPORT_BP=True, prefix=''):
    with torch.no_grad():
        mean_pix = torch.Tensor([x / 255.0 for x in mean_clr]).unsqueeze(-1).unsqueeze(-1).cuda()
        std_pix = torch.Tensor([x / 255.0 for x in std_clr]).unsqueeze(-1).unsqueeze(-1).cuda()

        # denormalize and prepare the original query images
        if SHOW_QUERY_BP or SHOW_VM:
            orig = inputs[1][bb]
            orig = [x * std_pix + mean_pix for x in orig]

        if SHOW_SUPPORT_BP:
            sups, sup_bps = [], []
            L = support_labels[bb].cpu().numpy()
            for i_q in range(nQ):
                ix = np.where(L == c[bb, i_q].item())[0]
                sup = inputs[0][bb, ix[0]]
                sup = sup * std_pix + mean_pix
                sups.append(sup)

                sup_bp = bp_support[bb, i_q, ix[0]].detach().cpu().numpy()  # c[bb, i_q]
                sup_bp = torch.Tensor(cm.jet(sup_bp / sup_bp.max())[:, :, :-1]).permute(2, 0, 1)
                sup_bps.append(sup_bp)
            sup_bps = blend(sup_bps, sups, normalize=False)
            gridBP_sup = tv.utils.make_grid(sup_bps, nrow=6, normalize=False, pad_value=0.5)
            tv.utils.save_image(gridBP_sup, os.path.join(img_save_path, prefix + 'bp_grid_support.png'))
            if tb is not None:
                tb.add_image(tb_prefix + prefix + 'bps (support)', gridBP_sup, 0)

        # back-projection maps (query)
        if SHOW_QUERY_BP:
            bpms = []
            for q in range(nQ):
                if len(bp_query.shape) == 5:
                    bpp = bp_query[bb, q, int(c[bb, q].item())].detach().cpu().numpy()
                else:
                    bpp = bp_query[bb, q].detach().cpu().numpy()
                bpp = torch.Tensor(cm.jet(bpp / bpp.max())[:, :, :-1]).permute(2, 0, 1)
                bpms.append(bpp)
            bpms = blend(bpms, orig, normalize=False)
            gridBP = tv.utils.make_grid(bpms, nrow=6, normalize=False, pad_value=0.5)
            tv.utils.save_image(gridBP, os.path.join(img_save_path, prefix + 'bp_grid_query.png'))
            if tb is not None:
                tb.add_image(tb_prefix + prefix + 'bps (query)', gridBP, 0)

        # voting maps
        if SHOW_VM:
            vm = vm.view(b, nQ, nC, mxy, mxx)
            cvms = [vm[bb, j, int(c[bb, j].item()), int(y / 2) - 1:-int(y / 2), int(x / 2) - 1:-int(x / 2)].cpu() for j
                    in range(nQ)]
            cvms = [torch.Tensor(cm.jet(x / (x.max() + 1e-7))[:, :, :-1]).permute(2, 0, 1) for x in cvms]
            pp = 1  # 7
            cvms = [x.pow(pp) / (x.pow(pp).max() + 1e-5) for x in cvms]
            cvms = blend(cvms, orig, normalize=False)
            gridVM = tv.utils.make_grid(cvms, nrow=6, normalize=False, pad_value=0.5)
            tv.utils.save_image(gridVM, os.path.join(img_save_path, prefix + 'vm_grid.png'))
            if tb is not None:
                tb.add_image(tb_prefix + prefix + 'vms', gridVM, 0)

def report_query_images_noGT(inputs, mean_clr, std_clr, c,bboxes, bp_query,pred_classses_set,
                        gt_boxes_set,gt_classes_set,
                        nQ, img_save_path, tb=None, tb_prefix='', prefix=''):
    bb=0

    with torch.no_grad():
        mean_pix = torch.Tensor([x / 255.0 for x in mean_clr]).unsqueeze(-1).unsqueeze(-1).cuda()
        std_pix = torch.Tensor([x / 255.0 for x in std_clr]).unsqueeze(-1).unsqueeze(-1).cuda()

        # denormalize and prepare the original query images
        orig = inputs[1][bb]
        orig = [x * std_pix + mean_pix for x in orig]

        # back-projection maps (query)
        bpms = []
        for q in range(nQ):
            if len(bp_query.shape) == 5:
                bpp = bp_query[bb, q, int(c[bb, q].item())].detach().cpu().numpy()
            else:
                bpp = bp_query[bb, q].detach().cpu().numpy()
            bpp = torch.Tensor(cm.jet(bpp / bpp.max())[:, :, :-1]).permute(2, 0, 1)
            #bpp=add_bboxes(bbp,bboxes[q])
            bpms.append(bpp)

            img = orig[q].cpu()
            bboxes_q = bboxes[q]
            pred_classses_q = pred_classses_set[q]
            gt_classes_q = gt_classes_set[q]
            gt_boxes_q = gt_boxes_set[q]
            img = img.cpu()
            img_b = np.ascontiguousarray(np.asarray(img.permute(2, 1, 0)),dtype=np.uint8)
            for bbox,pred_class,gt_bbox,gt_class  in zip(bboxes_q,pred_classses_q,gt_boxes_q, gt_classes_q):
                try:
                    img_b = cv.rectangle(img_b, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (205, 65, 0), 2).get()
                except:
                    img_b = cv.rectangle(img_b, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (205, 65, 0), 2)

            try:
                img_b = torch.from_numpy(img_b).permute(2, 1, 0)
            except:
                img_b = torch.from_numpy(img_b.get()).permute(2, 1, 0)

            q_blend = blend([bpp], [img_b.cuda()], normalize=False)
            #q_blend[0] = q_blend[0].transpose(2,1)
            tv.utils.save_image(q_blend, os.path.join(img_save_path, prefix + 'query_bbox_{}.png'.format(q)))

            orig[q] = img_b.cuda()
        batch_blend = blend(bpms, orig, normalize=False)
        gridBP = tv.utils.make_grid(batch_blend, nrow=6, normalize=False, pad_value=0.5)
        tv.utils.save_image(gridBP, os.path.join(img_save_path, prefix + 'bp_bbox_query.png'))
        # if tb is not None:
        #     tb.add_image(tb_prefix + prefix + 'bps (query)', gridBP, 0)

def report_query_images(inputs, mean_clr, std_clr, c,bboxes, bp_query,pred_classses_set,
                        gt_boxes_set,gt_classes_set,
                        nQ, img_save_path, tb=None, tb_prefix='', prefix=''):
    bb=0

    with torch.no_grad():
        mean_pix = torch.Tensor([x / 255.0 for x in mean_clr]).unsqueeze(-1).unsqueeze(-1).cuda()
        std_pix = torch.Tensor([x / 255.0 for x in std_clr]).unsqueeze(-1).unsqueeze(-1).cuda()

        # denormalize and prepare the original query images
        orig = inputs[1][bb]
        orig = [x * std_pix + mean_pix for x in orig]

        # back-projection maps (query)
        bpms = []
        for q in range(nQ):
            if len(bp_query.shape) == 5:
                bpp = bp_query[bb, q, int(c[bb, q].item())].detach().cpu().numpy()
            else:
                bpp = bp_query[bb, q].detach().cpu().numpy()
            bpp = torch.Tensor(cm.jet(bpp / bpp.max())[:, :, :-1]).permute(2, 0, 1)
            #bpp=add_bboxes(bbp,bboxes[q])
            bpms.append(bpp)
        for q in range(nQ):# for idx in range(len(orig)):
            img = orig[q].cpu()
            bboxes_q = bboxes[q]
            pred_classses_q = pred_classses_set[q]
            gt_classes_q = gt_classes_set[q]
            gt_boxes_q = gt_boxes_set[q]
            img = img.cpu()
            img_b = np.asarray(img.permute(2, 1, 0))
            for bbox,pred_class,gt_bbox,gt_class  in zip(bboxes_q,pred_classses_q,gt_boxes_q, gt_classes_q):
                try:
                    img_b = cv.rectangle(img_b, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (255, 0, 0), 1).get()
                except:
                    img_b = cv.rectangle(img_b, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (255, 0, 0), 1)
                #img_b = cv.putText(img_b,str(pred_class),(bbox[1], bbox[0]),cv.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),2)
                img_b = cv.putText(img_b.transpose(1,0,2), str(pred_class), (bbox[1], bbox[0]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2).get().transpose(1,0,2)

                img_b = cv.rectangle(img_b, (gt_bbox[1], gt_bbox[0]), (gt_bbox[3], gt_bbox[2]), (0,255, 0), 1).get()
                img_b = cv.putText(img_b.transpose(1,0,2),str(gt_class),(gt_bbox[1], gt_bbox[0]),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255, 0),2).get().transpose(1,0,2)
                IoU =get_IoU(bbox,gt_bbox)
                img_b = cv.putText(img_b.transpose(1,0,2), 'IoU={:.2f}'.format(IoU), (0, 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2).transpose(1,0,2)

            img_b = torch.from_numpy(img_b).permute(2, 1, 0)
            orig[q] = img_b.cuda()

        bpms = blend(bpms, orig, normalize=False)
        gridBP = tv.utils.make_grid(bpms, nrow=6, normalize=False, pad_value=0.5)
        tv.utils.save_image(gridBP, os.path.join(img_save_path, prefix + 'bp_bbox_query.png'))
        if tb is not None:
            tb.add_image(tb_prefix + prefix + 'bps (query)', gridBP, 0)

def gen_bbox_from_maps(sup_bps, images, img_save_path, naming):
    import cv2
    from bbox_generator import gen_bbox_img

    # size_upsample = (256, 256)
    # output_cam.append(cv2.resize(cam_img, size_upsample))
    # img = cv2.imread('test.jpg')
    # height, width, _ = img.shape
    # heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    # result = heatmap * 0.3 + img * 0.5
    # cv2.imwrite('CAM.jpg', result)
    # curHeatMap = cv2.resize(im2double(curCAMmapLarge_crops), (256, 256))  # this line is not doing much
    # curHeatMap = im2double(curHeatMap)

    # create and save heat maps
    heat_map_paths = []
    for output_img_index, sup_bp in enumerate(sup_bps):
        heat_map_save_path = os.path.join(img_save_path, str(naming) + '_heat_map_' + str(output_img_index) + '.png')
        img = images[output_img_index]
        cam_img = np.uint8(255 * sup_bp)
        img = np.uint8(255 * img)
        img = img.transpose((1,2,0))
        cam_img = cam_img.transpose((1,2,0))
        # cam_img = cv2.resize(cam_img, size_upsample)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(cam_img, (width, height)), cv2.COLORMAP_JET)
        # heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
        result = heatmap * 0.2 + img * 0.7
        # result = heatmap * 0.3
        heat_map_paths += [heat_map_save_path]
        cv2.imwrite(heat_map_save_path, result)
    # generate and save bbox's
    bbox_threshold = [20, 100, 110]  # parameters for the bbox generator
    curParaThreshold = str(bbox_threshold[0]) + ' ' + str(bbox_threshold[1]) + ' ' + str(bbox_threshold[2]) + ' '
    for output_bbox_index, heat_map_path in enumerate(heat_map_paths):
        output_bbox_path = os.path.join(img_save_path, str(naming) + '_bbox_' + str(output_bbox_index) + '.txt')
        os.system("CAM_Python_master/bboxgenerator/./dt_box " + heat_map_path + ' ' +
                  curParaThreshold + ' ' + output_bbox_path)
        output_bbox_img_path = os.path.join(img_save_path, str(naming) + '_bbox_' + str(output_bbox_index) + '.png')
        # img = images[output_bbox_index]
        # img = np.uint8(255 * img)
        # img = img.transpose((1, 2, 0))
        gen_bbox_img(heat_map_path, output_bbox_path, output_bbox_img_path)



# comb_type is 'avg', or 'max'
def support2class( bp_query_, vm_, support_labels, b, nQ, nS, nC, y, x, bp_query_normalize = True, comb_type = 'avg' ):
    support_labels_one_hot = one_hot(support_labels.view(b * nS), nC)
    support_labels_one_hot = support_labels_one_hot.view(b, nS, nC)  # b, nS, nC

    bp_query = None

    if comb_type == 'avg':
        # transform vm and bp_query into class based and not support instance based
        if bp_query_ is not None:
            bp_query = torch.matmul(  # TBD: this accumulation uses different max score location for every support
                support_labels_one_hot.permute(0, 2, 1).unsqueeze(1), # b, 1, nC, nS
                bp_query_.view(b, nQ, nS, -1) # b, nQ, nS, y*x
            ).view(b, nQ, nC, y, x) # b, nQ, nC, y, x

        vm = torch.matmul(
            support_labels_one_hot.permute(0, 2, 1).unsqueeze(1),
            vm_.view(b, nQ, nS, -1)
        )  # b, nQ, nC, y*x

    elif comb_type == 'max':
        if bp_query_ is not None:
            bp_query = torch.zeros( b, nQ, nC, y * x ).cuda()
        vm = torch.zeros( b, nQ, nC, vm_.shape[-2] *  vm_.shape[-1]).cuda()
        support_labels_one_hot_ = support_labels_one_hot.permute(0, 2, 1).unsqueeze(1)  # b, 1, nC, nS
        if bp_query_ is not None:
            bp_query_ = bp_query_.view(b, nQ, nS, -1)
        vm_ = vm_.view(b, nQ, nS, -1)
        for iS in range(nS):
            if bp_query_ is not None:
                bp_query = torch.max(
                    bp_query,
                    torch.matmul(  # TBD: this accumulation uses different max score location for every support
                        support_labels_one_hot_[:, :, :, [iS]],
                        bp_query_[:, :, [iS], :]  # b, nQ, nS, y*x
                    )
                )
            vm = torch.max(
                vm,
                torch.matmul(
                    support_labels_one_hot_[:, :, :, [iS]],
                    vm_[:, :, [iS], :]
                )
            )
        if bp_query_ is not None:
            bp_query = bp_query.view(b, nQ, nC, y, x) # b, nQ, nC, y, x

    elif comb_type == 'max_JS':

        bp_query = torch.zeros( b, nQ, nC, y * x ).cuda()
        vm = torch.zeros( b, nQ, nC, vm_.shape[-2] *  vm_.shape[-1]).cuda()
        support_labels_one_hot_ = support_labels_one_hot.permute(0, 2, 1).unsqueeze(1)  # b, 1, nC, nS
        bp_query_ = bp_query_.view(b, nQ, nS, -1)
        vm_ = vm_.view(b, nQ, nS, -1)
        for iC in range(nC):
            zho = bp_query_.unsqueeze(2)*support_labels_one_hot_[:, :,[iC],:].unsqueeze(4)
            bp_query[:,:,[iC],:] = torch.max(zho.squeeze(2),dim=2).values.unsqueeze(2)
            rho = vm_.unsqueeze(2)*support_labels_one_hot_[:, :,[iC],:].unsqueeze(4)
            vm[:,:,[iC],:] = torch.max(rho.squeeze(2),dim=2).values.unsqueeze(2)

        if bp_query_ is not None:
            bp_query = bp_query.view(b, nQ, nC, y, x) # b, nQ, nC, y, x


    else:
        raise Exception('unsupported comb_type: {}'.format(comb_type))


    # normalize after accumulation
    if bp_query_normalize and (bp_query_ is not None):
        bp_query /= (bp_query.max(dim=-2, keepdim=True).values.max(dim=-1, keepdim=True).values + 1e-7)

    return bp_query, vm, support_labels_one_hot

def back_project(vm, accp, c, szYX, kernel_size, max_only = True, bp_normalize = True, bp_sigma=2.0):
    global targX, targY
    b, nQ, nS, mxy, mxx = vm.shape

    bp, bp_support = None, None

    vm = vm.view(b * nQ * nS, mxy * mxx)

    if max_only:
        b_ix, q_ix = torch.meshgrid(
            torch.arange(0, b, dtype=torch.long),
            torch.arange(0, nQ, dtype=torch.long)
        )
        c_ix = b_ix.contiguous().view(-1).cuda() * b * nS + q_ix.contiguous().view(-1).cuda() * nS + c.view(-1)

    if max_only:
        predYX = torch.max(vm[c_ix, :], dim=1)
        predVal = predYX.values # b * nQ
        predYX = predYX.indices # b * nQ
    else:
        predYX = torch.max(vm, dim=1)
        predVal = predYX.values  # b * nQ * nS
        predYX = predYX.indices  # b * nQ * nS
    predX_ = predYX.fmod(mxx)
    predY_ = (predYX - predX_) // mxx

    predX = predX_.unsqueeze(-1).unsqueeze(-1).type(torch.cuda.FloatTensor)
    predY = predY_.unsqueeze(-1).unsqueeze(-1).type(torch.cuda.FloatTensor)

    # targX and targY are originally b*nQ*nS, x * y , x * y
    if max_only:
        targX_ = targX[c_ix] # b*nQ, x*y, x*y
        targY_ = targY[c_ix]  # b*nQ, x*y, x*y
    else:
        targX_ = targX # b*nQ*nS, x*y, x*y
        targY_ = targY # b*nQ*nS, x*y, x*y

    #std = 2.0
    bp_votes = torch.exp(
        -0.5 *
        (
                (targX_.type(torch.cuda.FloatTensor) - predX).pow(2) +
                (targY_.type(torch.cuda.FloatTensor) - predY).pow(2)
        ) / (bp_sigma * bp_sigma)
    )

    if max_only:
        bp_votes *= accp.view(b*nQ*nS,bp_votes.shape[-2],bp_votes.shape[-1])[c_ix,:,:]
        bp = bp_votes.sum(dim=2).view(b * nQ, 1, szYX[0], szYX[1])
    else:
        bp_votes *= accp.view(b * nQ * nS, bp_votes.shape[-2], bp_votes.shape[-1])
        bp = bp_votes.sum(dim=2).view(b * nQ * nS, 1, szYX[0], szYX[1])

    bp = F.conv2d(bp, kernel, bias=None, stride=1,
                  padding=int((kernel_size[0] - 1) / 2), dilation=1, groups=1)

    if max_only:
        bp = bp.view(b, nQ, szYX[0], szYX[1])
    else:
        bp = bp.view(b, nQ, nS, szYX[0], szYX[1])

    if bp_normalize:
        bp /= (bp.max(dim=-2, keepdim=True).values.max(dim=-1, keepdim=True).values + 1e-7)

    if max_only:
        bp_support = bp_votes.sum(dim=1).view(b*nQ, 1, szYX[0], szYX[1])
    else:
        bp_support = bp_votes.sum(dim=1).view(b * nQ * nS, 1, szYX[0], szYX[1])

    bp_support = F.conv2d(bp_support, kernel, bias=None, stride=1,
                  padding=int((kernel_size[0] - 1) / 2), dilation=1, groups=1)

    if max_only:
        bp_support = bp_support.view(b, nQ, szYX[0], szYX[1])
        bp_support_ = bp_support.unsqueeze(dim=2).repeat(1, 1, nS, 1, 1)
    else:
        bp_support_ = bp_support.view(b, nQ, nS, szYX[0], szYX[1])

    bp_support_ /= (bp_support_.max(dim=-2, keepdim=True).values.max(dim=-1, keepdim=True).values + 1e-7)

    if max_only:
        predY_, predX_, predVal = predY_.view(b, nQ), predX_.view(b, nQ), predVal.view(b, nQ)
    else:
        predY_, predX_, predVal = predY_.view(b, nQ, nS), predX_.view(b, nQ, nS), predVal.view(b, nQ, nS)

    return bp, bp_support_, predY_, predX_, predVal

def blend(maskTensors, imageTensors, alpha=0.7, thresh=0.1, normalize = True):
    # resize the masks to fit the images
    toPIL = tv.transforms.ToPILImage()
    toTensor = tv.transforms.ToTensor()
    rsz = tv.transforms.Resize(imageTensors[0].shape[-2:], 2)
    maskTensors = [toTensor(rsz(toPIL(x.cpu()))).cuda() for x in maskTensors]

    #normalize and create alphas
    eps = 1e-7
    if normalize:
        alphas = [(m/(m.max()+eps)) for m in maskTensors]
    else:
        alphas = maskTensors
    alphas = [alpha * a * (a >= thresh).type(torch.FloatTensor).cuda() for a in alphas]
    res = [
        m * a + img * (1.0 - a)
        for a, m, img in zip(alphas, maskTensors, imageTensors)
    ]
    return res

def ProtoNetHead(query, support, support_labels, n_way, n_shot, normalize=True):
    """
    Constructs the prototype representation of each class(=mean of support vectors of each class) and 
    returns the classification score (=L2 distance to each class prototype) on the query set.
    
    This model is the classification head described in:
    Prototypical Networks for Few-shot Learning
    (Snell et al., NIPS 2017).
    
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)
    
    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    
    # From:
    # https://github.com/gidariss/FewShotWithoutForgetting/blob/master/architectures/PrototypicalNetworksHead.py
    #************************* Compute Prototypes **************************
    labels_train_transposed = support_labels_one_hot.transpose(1,2)
    # Batch matrix multiplication:
    #   prototypes = labels_train_transposed * features_train ==>
    #   [batch_size x nKnovel x num_channels] =
    #       [batch_size x nKnovel x num_train_examples] * [batch_size * num_train_examples * num_channels]
    prototypes = torch.bmm(labels_train_transposed, support)
    # Divide with the number of examples per novel category.
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
    )

    # Distance Matrix Vectorization Trick
    AB = computeGramMatrix(query, prototypes)
    AA = (query * query).sum(dim=2, keepdim=True)
    BB = (prototypes * prototypes).sum(dim=2, keepdim=True).reshape(tasks_per_batch, 1, n_way)
    logits = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
    logits = -logits
    
    if normalize:
        logits = logits / d

    return logits

def MetaOptNetHead_SVM_CS(query, support, support_labels, n_way, n_shot, C_reg=0.1, double_precision=False, maxIter=15):
    """
    Fits the support set with multi-class SVM and 
    returns the classification score on the query set.
    
    This is the multi-class SVM presented in:
    On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines
    (Crammer and Singer, Journal of Machine Learning Research 2001).

    This model is the classification head that we use for the final version.
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      C_reg: a scalar. Represents the cost parameter C in SVM.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot

    #Here we solve the dual problem:
    #Note that the classes are indexed by m & samples are indexed by i.
    #min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
    #s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i

    #where w_m(\alpha) = \sum_i \alpha^m_i x_i,
    #and C^m_i = C if m  = y_i,
    #C^m_i = 0 if m != y_i.
    #This borrows the notation of liblinear.
    
    #\alpha is an (n_support, n_way) matrix
    kernel_matrix = computeGramMatrix(support, support)

    id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).cuda()
    block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)
    #This seems to help avoid PSD error from the QP solver.
    block_kernel_matrix += 1.0 * torch.eye(n_way*n_support).expand(tasks_per_batch, n_way*n_support, n_way*n_support).cuda()
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way) # (tasks_per_batch * n_support, n_support)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, n_support * n_way)
    
    G = block_kernel_matrix
    e = -1.0 * support_labels_one_hot
    #print (G.size())
    #This part is for the inequality constraints:
    #\alpha^m_i <= C^m_i \forall m,i
    #where C^m_i = C if m  = y_i,
    #C^m_i = 0 if m != y_i.
    id_matrix_1 = torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support, n_way * n_support)
    C = Variable(id_matrix_1)
    h = Variable(C_reg * support_labels_one_hot)
    #print (C.size(), h.size())
    #This part is for the equality constraints:
    #\sum_m \alpha^m_i=0 \forall i
    id_matrix_2 = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

    A = Variable(batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, n_way).cuda()))
    b = Variable(torch.zeros(tasks_per_batch, n_support))
    #print (A.size(), b.size())
    if double_precision:
        G, e, C, h, A, b = [x.double().cuda() for x in [G, e, C, h, A, b]]
    else:
        G, e, C, h, A, b = [x.float().cuda() for x in [G, e, C, h, A, b]]

    # Solve the following QP to fit SVM:
    #        \hat z =   argmin_z 1/2 z^T G z + e^T z
    #                 subject to Cz <= h
    # We use detach() to prevent backpropagation to fixed variables.
    qp_sol = QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())

    # Compute the classification score.
    compatibility = computeGramMatrix(support, query)
    compatibility = compatibility.float()
    compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
    qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
    logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
    logits = logits * compatibility
    logits = torch.sum(logits, 1)

    return logits

def MetaOptNetHead_SVM_WW(query, support, support_labels, n_way, n_shot, C_reg=0.00001, double_precision=False):
    """
    Fits the support set with multi-class SVM and 
    returns the classification score on the query set.
    
    This is the multi-class SVM presented in:
    Support Vector Machines for Multi Class Pattern Recognition
    (Weston and Watkins, ESANN 1999).
    
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      C_reg: a scalar. Represents the cost parameter C in SVM.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    """
    Fits the support set with multi-class SVM and 
    returns the classification score on the query set.
    
    This is the multi-class SVM presented in:
    Support Vector Machines for Multi Class Pattern Recognition
    (Weston and Watkins, ESANN 1999).
    
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      C_reg: a scalar. Represents the cost parameter C in SVM.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot

    #In theory, \alpha is an (n_support, n_way) matrix
    #NOTE: In this implementation, we solve for a flattened vector of size (n_way*n_support)
    #In order to turn it into a matrix, you must first reshape it into an (n_way, n_support) matrix
    #then transpose it, resulting in (n_support, n_way) matrix
    kernel_matrix = computeGramMatrix(support, support) + torch.ones(tasks_per_batch, n_support, n_support).cuda()
    
    id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).cuda()
    block_kernel_matrix = batched_kronecker(id_matrix_0, kernel_matrix)
    
    kernel_matrix_mask_x = support_labels.reshape(tasks_per_batch, n_support, 1).expand(tasks_per_batch, n_support, n_support)
    kernel_matrix_mask_y = support_labels.reshape(tasks_per_batch, 1, n_support).expand(tasks_per_batch, n_support, n_support)
    kernel_matrix_mask = (kernel_matrix_mask_x == kernel_matrix_mask_y).float()
    
    block_kernel_matrix_inter = kernel_matrix_mask * kernel_matrix
    block_kernel_matrix += block_kernel_matrix_inter.repeat(1, n_way, n_way)
    
    kernel_matrix_mask_second_term = support_labels.reshape(tasks_per_batch, n_support, 1).expand(tasks_per_batch, n_support, n_support * n_way)
    kernel_matrix_mask_second_term = kernel_matrix_mask_second_term == torch.arange(n_way).long().repeat(n_support).reshape(n_support, n_way).transpose(1, 0).reshape(1, -1).repeat(n_support, 1).cuda()
    kernel_matrix_mask_second_term = kernel_matrix_mask_second_term.float()
    
    block_kernel_matrix -= (2.0 - 1e-4) * (kernel_matrix_mask_second_term * kernel_matrix.repeat(1, 1, n_way)).repeat(1, n_way, 1)

    Y_support = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    Y_support = Y_support.view(tasks_per_batch, n_support, n_way)
    Y_support = Y_support.transpose(1, 2)   # (tasks_per_batch, n_way, n_support)
    Y_support = Y_support.reshape(tasks_per_batch, n_way * n_support)
    
    G = block_kernel_matrix

    e = -2.0 * torch.ones(tasks_per_batch, n_way * n_support)
    id_matrix = torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support, n_way * n_support)
            
    C_mat = C_reg * torch.ones(tasks_per_batch, n_way * n_support).cuda() - C_reg * Y_support

    C = Variable(torch.cat((id_matrix, -id_matrix), 1))
    #C = Variable(torch.cat((id_matrix_masked, -id_matrix_masked), 1))
    zer = torch.zeros(tasks_per_batch, n_way * n_support).cuda()
    
    h = Variable(torch.cat((C_mat, zer), 1))
    
    dummy = Variable(torch.Tensor()).cuda()      # We want to ignore the equality constraint.

    if double_precision:
        G, e, C, h = [x.double().cuda() for x in [G, e, C, h]]
    else:
        G, e, C, h = [x.cuda() for x in [G, e, C, h]]

    # Solve the following QP to fit SVM:
    #        \hat z =   argmin_z 1/2 z^T G z + e^T z
    #                 subject to Cz <= h
    # We use detach() to prevent backpropagation to fixed variables.
    #qp_sol = QPFunction(verbose=False)(G, e.detach(), C.detach(), h.detach(), dummy.detach(), dummy.detach())
    qp_sol = QPFunction(verbose=False)(G, e, C, h, dummy.detach(), dummy.detach())

    # Compute the classification score.
    compatibility = computeGramMatrix(support, query) + torch.ones(tasks_per_batch, n_support, n_query).cuda()
    compatibility = compatibility.float()
    compatibility = compatibility.unsqueeze(1).expand(tasks_per_batch, n_way, n_support, n_query)
    qp_sol = qp_sol.float()
    qp_sol = qp_sol.reshape(tasks_per_batch, n_way, n_support)
    A_i = torch.sum(qp_sol, 1)   # (tasks_per_batch, n_support)
    A_i = A_i.unsqueeze(1).expand(tasks_per_batch, n_way, n_support)
    qp_sol = qp_sol.float().unsqueeze(3).expand(tasks_per_batch, n_way, n_support, n_query)
    Y_support_reshaped = Y_support.reshape(tasks_per_batch, n_way, n_support)
    Y_support_reshaped = A_i * Y_support_reshaped
    Y_support_reshaped = Y_support_reshaped.unsqueeze(3).expand(tasks_per_batch, n_way, n_support, n_query)
    logits = (Y_support_reshaped - qp_sol) * compatibility

    logits = torch.sum(logits, 2)

    return logits.transpose(1, 2)

class ClassificationHead(nn.Module):
    def __init__(self, base_learner='MetaOptNet', enable_scale=True, split_scales = False):
        super(ClassificationHead, self).__init__()
        if ('SVM-CS' in base_learner):
            self.head = MetaOptNetHead_SVM_CS
        elif ('Ridge' in base_learner):
            self.head = MetaOptNetHead_Ridge
        elif ('R2D2' in base_learner):
            self.head = R2D2Head
        elif ('Proto' in base_learner):
            self.head = ProtoNetHead
        elif ('Star' in base_learner):
            self.head = StarNetHead
        elif ('SVM-He' in base_learner):
            self.head = MetaOptNetHead_SVM_He
        elif ('SVM-WW' in base_learner):
            self.head = MetaOptNetHead_SVM_WW
        else:
            print ("Cannot recognize the base learner type")
            assert(False)
        
        # Add a learnable scale
        self.enable_scale = enable_scale
        self.split_scales = split_scales
        if not self.split_scales:
            self.scale = nn.Parameter(torch.FloatTensor([1.0]))
        else:
            # self.scale = [ nn.Parameter(torch.FloatTensor([1.0])).cuda(), nn.Parameter(torch.FloatTensor([1.0])).cuda() ]
            self.scale1 = nn.Parameter(torch.FloatTensor([1.0]))
            self.scale2 = nn.Parameter(torch.FloatTensor([1.0]))
            self.scale = [ self.scale1, self.scale2 ]

    def forward(self, query, support, support_labels, n_way, n_shot, *args, **kwargs):
        if self.enable_scale:
            logits = self.head(query, support, support_labels, n_way, n_shot, *args, **kwargs)
            if type(logits) is tuple:
                prox_query_set= logits[3]
                class_indices = logits[2]
                bp_query = logits[1]
                logits = logits[0]
            else:
                bp_query = None
                class_indices = None
                prox_query_set = None

            if type(logits) is list:
                if self.split_scales:
                    ret = [ s * l for l, s in zip(logits, self.scale)]
                else:
                    ret = [ self.scale * l for l in logits]
            else:
                if self.split_scales:
                    ret = self.scale[0] * logits
                else:
                    ret = self.scale * logits
        else:
            ret = self.head(query, support, support_labels, n_way, n_shot, *args, **kwargs)

        return ret, bp_query, class_indices,prox_query_set
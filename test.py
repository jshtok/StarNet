# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.autograd import Variable

from tqdm import tqdm

from models.protonet_embedding import ProtoNetEmbedding
from models.R2D2_embedding import R2D2Embedding
from models.ResNet12_embedding import resnet12

from models.classification_heads import ClassificationHead

from utils import pprint, set_gpu, Timer, count_accuracy, log, get_model

import numpy as np
import os
import pydevd_pycharm
from utils import finetune

# def get_model(options):
#     # Choose the embedding network
#     if options.network == 'ProtoNet':
#         network = ProtoNetEmbedding().cuda()
#     elif options.network == 'R2D2':
#         network = R2D2Embedding().cuda()
#     elif options.network == 'ResNet':
#         if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
#             network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5).cuda()
#             network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
#         else:
#             network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2).cuda()
#     else:
#         print ("Cannot recognize the network type")
#         assert(False)
#
#     # Choose the classification head
#     if opt.head == 'ProtoNet':
#         cls_head = ClassificationHead(base_learner='ProtoNet').cuda()
#     elif opt.head == 'Ridge':
#         cls_head = ClassificationHead(base_learner='Ridge').cuda()
#     elif opt.head == 'R2D2':
#         cls_head = ClassificationHead(base_learner='R2D2').cuda()
#     elif opt.head == 'SVM':
#         cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
#     else:
#         print ("Cannot recognize the classification head type")
#         assert(False)
#
#     return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_test = MiniImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_test = tieredImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_test = CIFAR_FS(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'FC100':
        from data.FC100 import FC100, FewShotDataloader
        dataset_test = FC100(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'cub':
        from data.cub import Cub, FewShotDataloader
        dataset_test = Cub(phase='test')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_test, data_loader)

def setup_debug(args):
    if args.debug:
        pydevd_pycharm.settrace(args.debug_addr, port=args.debug_port, stdoutToServer=True, stderrToServer=True, suspend=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='system_set')
    parser.add_argument('--load', default='./experiments/exp_1/best_model.pth',
                            help='path of the checkpoint file')
    parser.add_argument('--episode', type=int, default=1000,
                            help='number of episodes to test')
    parser.add_argument('--way', type=int, default=5,
                            help='number of classes in one test episode')
    parser.add_argument('--shot', type=int, default=1,
                            help='number of support examples per training class')
    parser.add_argument('--query', type=int, default=15,
                            help='number of query examples per training class')
    parser.add_argument('--network', type=str, default='ProtoNet',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='ProtoNet',
                            help='choose which embedding network to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--debug', action='store_true', help='')
    parser.add_argument('--debug_port', default=1327 , type=int, help='for debug')
    parser.add_argument('--debug_addr', default='9.148.202.199', type=str, help='for debug')

    parser.add_argument('--two_stage', type=int, default=1,
                        help='one-stage vs two-stage model')
    parser.add_argument('--normalize', type=int, default=1,
                        help='normalize the features or not')
    parser.add_argument('--allow_stage_cross_talk', type=int, default=0,
                        help='will there be gradients flowing from the second stage to the first stage using the bp_masks')
    parser.add_argument('--stage2_mode', type=int, default=3,
                        help='0 = without attention, 1 = query attention only, 2 = support attention only, 3 = both attentions')

    parser.add_argument('--vote_prob_nrm_type', type=int, default=0,
                        help='0 = without normalization, 1 = sum, 2 = max, 3 = sum current, 4 = max current')

    parser.add_argument('--split_head_scales', type=int, default=0,
                        help='0 = one scale for all, 1 = separate scale per stage')

    parser.add_argument('--finetune', type=int, default=0,
                        help='0 = no finetune, 1 = finetune')

    parser.add_argument('--multiple_hypotheses_perc', type=float, default=1.0,
                        help='NMS parameter, determines where to stop supressing, if 1.0 no multi-hypothese are used')

    parser.add_argument('--stage2_cosine', type=int, default=0,
                        help='0 = L2 sim, 1 = cosine sim')

    parser.add_argument('--support_comb_type', type=str, default='avg',
                        help='avg / max')

    parser.add_argument('--stage2_type', type=str, default='proto',
                        help='proto / star')

    opt = parser.parse_args()
    setup_debug(opt)
    (dataset_test, data_loader) = get_dataset(opt)

    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.way,
        nKbase=0,
        nExemplars=opt.shot, # num training examples per novel category
        nTestNovel=opt.query * opt.way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=1,
        epoch_size=opt.episode, # num of batches per epoch
    )

    set_gpu(opt.gpu)

    try:
        log_file_path = os.path.join(os.path.dirname(opt.load), "test_log.txt")
        log(log_file_path, str(vars(opt)))
    except:
        basename = os.path.basename(os.path.dirname(opt.load))
        log_file_path = os.path.join('./experiments', basename, "test_log.txt")
        log(log_file_path, str(vars(opt)))


    # Define the models
    (embedding_net, cls_head) = get_model(opt)
    
    # Load saved model checkpoints
    saved_models = torch.load(opt.load)

    if type(embedding_net) is list:
        loaded = saved_models['embedding']
        if type(loaded) is list:
            for iNet, net in enumerate(embedding_net):
                net.load_state_dict(loaded[iNet])
        else:
            embedding_net[0].load_state_dict(loaded)
    else:
        embedding_net.load_state_dict(saved_models['embedding'])

    # embedding_net.load_state_dict(saved_models['embedding'])
    # embedding_net.eval()
    if type(embedding_net) is list:
        for net in embedding_net:
            net.eval()
    else:
        embedding_net.eval()

    cls_head.load_state_dict(saved_models['head'], strict=False)
    cls_head.eval()

    if type(embedding_net) is list:
        embedding_net_st1 = embedding_net[0]
        embedding_net_st2 = embedding_net[1]  # TODO: support more then one
    else:
        embedding_net_st1 = embedding_net
        embedding_net_st2 = None
    
    # Evaluate on test set
    test_accuracies = []
    test_accuracies_parts = []
    for i, batch in enumerate(tqdm(dloader_test()), 1):
        data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

        if bool(opt.finetune):
            if type(embedding_net) is list:
                loaded = saved_models['embedding']
                for iNet, net in enumerate(embedding_net):
                    net.load_state_dict(loaded[iNet])
            else:
                embedding_net.load_state_dict(saved_models['embedding'])
            cls_head.load_state_dict(saved_models['head'], strict=False)
            cls_head.eval()

            if True:
                b, nQ, c, szy, szx = data_query.shape
                ft_data_query = data_query.unsqueeze(2).expand(-1,-1,opt.way,-1,-1,-1).contiguous().view(b, nQ * opt.way, c, szy, szx)
                ft_labels_query = torch.arange(0, opt.way).unsqueeze(0).expand(opt.way,-1).contiguous().view(1,-1).cuda()
                eps = 0.1
            else:
                ft_data_query = data_support
                ft_labels_query = labels_support
                eps = 0.0

            finetune(
                embedding_net, cls_head,
                data_support, labels_support, data_support, labels_support, #data_support, labels_support, data_query, labels_query,
                opt,
                learning_rate=0.001, num_iters=5,
                head=opt.head, train_query=1, train_way=opt.way, train_shot=1, episodes_per_batch=1,
                eps=eps
            )

        n_support = opt.way * opt.shot
        n_query = opt.way * opt.query
                
        emb_support = embedding_net_st1(data_support.reshape([-1] + list(data_support.shape[-3:])))
        if 'Star' in opt.head:
            emb_support = emb_support.reshape([1, n_support] + list(emb_support.shape[-3:]))
        else:
            emb_support = emb_support.reshape(1, n_support, -1)

        emb_query = embedding_net_st1(data_query.reshape([-1] + list(data_query.shape[-3:])))
        if 'Star' in opt.head:
            emb_query = emb_query.reshape([1, n_query] + list(emb_query.shape[-3:]))
        else:
            emb_query = emb_query.reshape(1, n_query, -1)

        s2_emb_support, s2_emb_query = None, None
        if embedding_net_st2 is not None:
            s2_emb_support = embedding_net_st2(data_support.reshape([-1] + list(data_support.shape[-3:])))
            if 'Star' in opt.head:
                s2_emb_support = s2_emb_support.reshape([1, n_support] + list(s2_emb_support.shape[-3:]))
            else:
                s2_emb_support = s2_emb_support.reshape(1, n_support, -1)

            s2_emb_query = embedding_net_st2(data_query.reshape([-1] + list(data_query.shape[-3:])))
            if 'Star' in opt.head:
                s2_emb_query = s2_emb_query.reshape([1, n_query] + list(s2_emb_query.shape[-3:]))
            else:
                s2_emb_query = s2_emb_query.reshape(1, n_query, -1)

        if opt.head == 'SVM':
            logit_query = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot, maxIter=3)
        else:
            logit_query = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot, inputs=(data_support, data_query), s2_query=s2_emb_query, s2_support=s2_emb_support, opt=opt)

        if type(logit_query) is tuple:
            bp_query = logit_query[1]
            logit_query = logit_query[0]
        else:
            bp_query = None

        if type(logit_query) is not list:
            logit_query = [logit_query]

        probs_query = [F.softmax(lq, dim=2) for lq in logit_query]
        logit_query = torch.stack(probs_query, dim=0).sum(dim=0) / float(len(logit_query))

        acc = count_accuracy(logit_query.reshape(-1, opt.way), labels_query.reshape(-1))
        acc_parts = [
            count_accuracy(pq.reshape(-1, opt.way), labels_query.reshape(-1))
            for pq in probs_query
        ]

        test_accuracies.append(acc.item())
        test_accuracies_parts.append([x.item() for x in acc_parts])
        
        avg = np.mean(np.array(test_accuracies))
        std = np.std(np.array(test_accuracies))
        ci95 = 1.96 * std / np.sqrt(i + 1)

        avg_parts = np.mean(np.array(test_accuracies_parts), axis=0)
        
        if i % 50 == 0:
            log(log_file_path,'Episode [{}/{}]:\t\t\tAccuracy: {:.2f} ± {:.2f} % ({:.2f} %, {})'\
                  .format(i, opt.episode, avg, ci95, acc, avg_parts))

    log(log_file_path,'Episode [{}/{}]:\t\t\tAccuracy: {:.2f} ± {:.2f} % ({:.2f} %, {})'\
          .format(i, opt.episode, avg, ci95, acc, avg_parts))

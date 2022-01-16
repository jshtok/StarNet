import os
import time
import psutil
import pprint
import torch

import torch.nn.functional as F
import numpy as np

from models.classification_heads import ClassificationHead
from models.R2D2_embedding import R2D2Embedding
from models.protonet_embedding import ProtoNetEmbedding
from models.ResNet12_embedding import resnet12

import torchvision.transforms as transforms

def assert_folder(folder):
    import os
    if not os.path.exists(folder):
        f_path, f_name = os.path.split(folder)
        if len(f_path)>0:
            assert_folder(f_path)
        os.mkdir(folder)
    return folder

def set_gpu(x):
    if x != 'system_set':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(x)
    else:
        x = os.environ['CUDA_VISIBLE_DEVICES']
        if x == '':
            x = os.environ['FULL_CUDA_VISIBLE_DEVICES']
    print('using gpu:', x)

def check_dir(path):
    '''
    Create directory if it does not exist.
        path:           Path of directory.
    '''
    if not os.path.exists(path):
        os.mkdir(path)

def count_accuracy(logits, label):
    pred = torch.argmax(logits, dim=1).view(-1)
    label = label.view(-1)
    accuracy = 100 * pred.eq(label).float().mean()
    return accuracy

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

def log(log_file_path, string):
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)

def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
    elif options.network == 'R2D2':
        network = R2D2Embedding().cuda()
    elif options.network == 'ResNet':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5).cuda()
            network = torch.nn.DataParallel(network)  # , device_ids=[0, 1, 2, 3])
        else:
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2).cuda()
    elif options.network == 'ResNet_star':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5, strides=[2, 2, 2, 1],
                               flatten=False).cuda()
            network = torch.nn.DataParallel(network)  # , device_ids=[0, 1, 2, 3])
        elif options.dataset == 'cub':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2, strides=[2, 2, 2, 1],
                               flatten=False).cuda()
        elif options.dataset == 'imagenet-loc' or options.dataset == 'imagenet-det' or options.dataset =='pascal_voc':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2, strides=[2, 2, 2, 1],
                               flatten=False).cuda()
            network = torch.nn.DataParallel(network)
        else:
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2, strides=[2, 2, 1, 1],
                               flatten=False).cuda()
    elif options.network == 'ResNet_star_hi':
        if options.dataset == 'imagenet-loc' or options.dataset == 'imagenet-det' or options.dataset =='pascal_voc':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2, strides=[2, 2, 2, 2],
                               flatten=False).cuda()
            network = torch.nn.DataParallel(network)

    elif options.network == 'ResNet_star_2stage':
        networks = []
        for iNet in range(2):
            if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
                network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5, strides=[2, 2, 2, 1],
                                   flatten=False).cuda()
                network = torch.nn.DataParallel(network)  # , device_ids=[0, 1, 2, 3])
            elif options.dataset == 'cub':
                network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2, strides=[2, 2, 2, 1],
                                   flatten=False).cuda()
            elif options.dataset == 'imagenet-loc' or options.dataset == 'imagenet-det'or options.dataset =='pascal_voc':
                network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2, strides=[2, 2, 2, 1],
                                   flatten=False).cuda()
                network = torch.nn.DataParallel(network)
            else:
                network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2, strides=[2, 2, 1, 1], # there are multiple options here
                                   flatten=False).cuda()
            networks.append(network)
        network = networks

    elif options.network == 'ResNet_star_2stage_hi':
        networks = []
        for iNet in range(2):
            if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
                network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5, strides=[2, 2, 2, 2],
                                   flatten=False).cuda()
                network = torch.nn.DataParallel(network)  # , device_ids=[0, 1, 2, 3])
            elif options.dataset == 'cub':
                network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2, strides=[2, 2, 2, 2],
                                   flatten=False).cuda()
            elif options.dataset == 'imagenet-loc' or options.dataset == 'imagenet-det' or options.dataset =='pascal_voc':
                network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2, strides=[2, 2, 2, 2],
                                   flatten=False).cuda()
                network = torch.nn.DataParallel(network)
            else:
                network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2, strides=[2, 2, 1, 2], # there are multiple options here
                                   flatten=False).cuda()
            networks.append(network)
        network = networks
    else:
        print("Cannot recognize the network type")
        assert (False)

    # Choose the classification head
    if options.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()
    if options.head == 'StarNet':
        cls_head = ClassificationHead(base_learner='StarNet', split_scales = bool(options.split_head_scales)).cuda()
    elif options.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif options.head == 'R2D2':
        cls_head = ClassificationHead(base_learner='R2D2').cuda()
    elif options.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    else:
        print("Cannot recognize the dataset type")
        assert (False)

    return (network, cls_head)

def print_memory_usage(log_file_path = None):
    process = psutil.Process(os.getpid())
    mem_usage_bytes = process.memory_info().rss
    gpu_mem_bytes = torch.cuda.memory_allocated()
    usage_str_cpu = 'CPU memory usage: {:.1f} GB / {:.1f} MB'.format(mem_usage_bytes / (1024 ** 3), mem_usage_bytes / (1024 ** 2))
    usage_str_gpu = 'GPU memory usage: {:.1f} GB / {:.1f} MB'.format(gpu_mem_bytes / (1024 ** 3), gpu_mem_bytes / (1024 ** 2))
    if log_file_path is None:
        print(usage_str_cpu)
        print(usage_str_gpu)
    else:
        log(log_file_path, usage_str_cpu)
        log(log_file_path, usage_str_gpu)

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
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies

def finetune(
        embedding_net, cls_head,
        data_support, labels_support, data_query, labels_query,
        opt,
        learning_rate, num_iters,
        head='StarNet', train_query=1, train_way=5, train_shot=1, episodes_per_batch = 1, eps = 0.1
):
    if type(embedding_net) is list:
        # pGroups = [{'params': net.parameters()} for net in embedding_net]
        # optimizer = torch.optim.SGD(
        #     pGroups + [{'params': cls_head.parameters()}],
        #     lr=learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True
        # )
        pGroups = [{'params': net.parameters()} for net in embedding_net[1:]]
        optimizer = torch.optim.SGD(
            pGroups,
            lr=learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True
        )
    else:
        optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                                     {'params': cls_head.parameters()}], lr=learning_rate, momentum=0.9, \
                                    weight_decay=5e-4, nesterov=True)

    lambda_epoch = lambda e: 1.0 if e < 20 else 1.0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    # no need as we actually need the model in eval mode
    # # move to train mode
    # if type(embedding_net) is list:
    #     for net in embedding_net:
    #         net.train()
    #     cls_head.train()
    # else:
    #     _, _ = [x.train() for x in (embedding_net, cls_head)]

    # we set to eval mode to keep the batch norms fixed
    if type(embedding_net) is list:
        for net in embedding_net:
            net.eval()
        cls_head.eval()
    else:
        _, _ = [x.eval() for x in (embedding_net, cls_head)]

    train_accuracies = []
    train_losses = []
    train_accuracies_parts = []
    train_losses_parts = []

    for i in range(1, num_iters + 1):
        # Train on the training split
        lr_scheduler.step()

        if type(embedding_net) is list:
            embedding_net_st1 = embedding_net[0]
            embedding_net_st2 = embedding_net[1]  # TODO: support more then one
        else:
            embedding_net_st1 = embedding_net
            embedding_net_st2 = None

        train_n_support = train_way * train_shot
        train_n_query = train_way * train_query

        emb_support = embedding_net_st1(data_support.reshape([-1] + list(data_support.shape[-3:])))
        if 'Star' in head:
            emb_support = emb_support.reshape(
                [episodes_per_batch, train_n_support] + list(emb_support.shape[-3:]))
        else:
            emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)

        emb_query = embedding_net_st1(data_query.reshape([-1] + list(data_query.shape[-3:])))
        if 'Star' in head:
            emb_query = emb_query.reshape([episodes_per_batch, train_n_query] + list(emb_query.shape[-3:]))
        else:
            emb_query = emb_query.reshape(episodes_per_batch, train_n_query, -1)

        s2_emb_support, s2_emb_query = None, None
        if embedding_net_st2 is not None:
            s2_emb_support = embedding_net_st2(data_support.reshape([-1] + list(data_support.shape[-3:])))
            if 'Star' in head:
                s2_emb_support = s2_emb_support.reshape(
                    [episodes_per_batch, train_n_support] + list(s2_emb_support.shape[-3:]))
            else:
                s2_emb_support = s2_emb_support.reshape(episodes_per_batch, train_n_support, -1)

            s2_emb_query = embedding_net_st2(data_query.reshape([-1] + list(data_query.shape[-3:])))
            if 'Star' in head:
                s2_emb_query = s2_emb_query.reshape(
                    [episodes_per_batch, train_n_query] + list(s2_emb_query.shape[-3:]))
            else:
                s2_emb_query = s2_emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)

        logit_query = cls_head(emb_query, emb_support, labels_support, train_way, train_shot,
                                   s2_query=s2_emb_query, s2_support=s2_emb_support, opt=opt)

        if type(logit_query) is not list:
            logit_query = [logit_query]

        smoothed_one_hot = one_hot(labels_query.reshape(-1), train_way)
        smoothed_one_hot = smoothed_one_hot * (1 - eps) + (1 - smoothed_one_hot) * eps / (train_way - 1)

        loss = 0.0
        loss_parts = [0.0] * len(logit_query)
        for iQ, lq in enumerate(logit_query):
            log_prb = F.log_softmax(lq.reshape(-1, train_way), dim=1)
            loss_ = -(smoothed_one_hot * log_prb).sum(dim=1)
            loss_parts[iQ] = loss_.mean()
            loss += loss_parts[iQ]

        probs_query = [F.softmax(lq, dim=2) for lq in logit_query]
        logit_query = torch.stack(probs_query, dim=0).sum(dim=0) / float(len(logit_query))

        acc = count_accuracy(logit_query.reshape(-1, train_way), labels_query.reshape(-1))
        acc_parts = [
            count_accuracy(pq.reshape(-1, train_way), labels_query.reshape(-1))
            for pq in probs_query
        ]

        train_accuracies.append(acc.item())
        train_losses.append(loss.item())
        train_accuracies_parts.append([x.item() for x in acc_parts])
        train_losses_parts.append([x.item() for x in loss_parts])

        train_acc_avg = np.mean(np.array(train_accuracies))
        loss_avg = np.mean(np.array(train_losses))
        train_acc_avg_parts = np.mean(np.array(train_accuracies_parts), axis=0)
        loss_avg_parts = np.mean(np.array(train_losses_parts), axis=0)

        if False:
            print(
                'Iter: [{}/{}]\tLoss: {:.4f} ({:.4f}, {})\tAccuracy: {:.2f}% ({:.2f}%, {})'.format(
                    i, num_iters, loss_avg, loss.item(), loss_avg_parts, train_acc_avg, acc,
                    train_acc_avg_parts)
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # # get back to test mode
    # if type(embedding_net) is list:
    #     for net in embedding_net:
    #         net.eval()
    #     cls_head.eval()
    # else:
    #     _, _ = [x.eval() for x in (embedding_net, cls_head)]

def convert_batch2vis(img_orig,mean_pix,std_pix):
    img = np.asarray(img_orig)
    img = img * std_pix.reshape(3, 1, 1)
    img = img + mean_pix.reshape(3, 1, 1)
    img = img[[2, 1, 0], :, :]
    img = img.astype(np.uint8)
    img = img.transpose([1, 2, 0])
    return img

def convert_vis2batch(img,mean_pix,std_pix):
    normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
    img = img.astype(np.float32)/255
    img = img.transpose([2, 0, 1])
    img = img[[2, 1, 0], :, :]
    img = torch.tensor(img)
    img = img.type(torch.FloatTensor)
    img = normalize(img)
    return img

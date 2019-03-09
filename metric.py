import numpy as np
import torch
from scipy import interpolate

from tqdm import tqdm

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp +fn==0) else float(tp) / float(tp +fn)
    fpr = 0 if (fp +tn==0) else float(fp) / float(fp +tn)

    acc = float(tp +tn ) /dist.shape[0]
    return tpr, fpr, acc

def calculate(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    return tp,fp,tn,fn

def ACER(threshold, dist, actual_issame):
    tp, fp, tn, fn = calculate(threshold, dist, actual_issame)

    apcer = fp / (tn*1.0 + fp*1.0)
    npcer = fn / (fn * 1.0 + tp * 1.0)
    acer = (apcer + npcer) / 2.0
    return acer,tp, fp, tn,fn

def TPR_FPR( dist, actual_issame, fpr_target = 0.001):
    # acer_min = 1.0
    # thres_min = 0.0
    # re = []

    # Positive
    # Rate(FPR):
    # FPR = FP / (FP + TN)

    # Positive
    # Rate(TPR):
    # TPR = TP / (TP + FN)

    thresholds = np.arange(0.0, 1.0, 0.001)
    nrof_thresholds = len(thresholds)

    fpr = np.zeros(nrof_thresholds)
    FPR = 0.0
    for threshold_idx, threshold in enumerate(thresholds):

        if threshold < 1.0:
            tp, fp, tn, fn = calculate(threshold, dist, actual_issame)
            FPR = fp / (fp*1.0 + tn*1.0)
            TPR = tp / (tp*1.0 + fn*1.0)

        fpr[threshold_idx] = FPR

    if np.max(fpr) >= fpr_target:
        f = interpolate.interp1d(np.asarray(fpr), thresholds, kind= 'slinear')
        threshold = f(fpr_target)
    else:
        threshold = 0.0

    tp, fp, tn, fn = calculate(threshold, dist, actual_issame)

    FPR = fp / (fp * 1.0 + tn * 1.0)
    TPR = tp / (tp * 1.0 + fn * 1.0)

    print(str(FPR)+' '+str(TPR))
    return FPR,TPR

import torch.nn.functional as F
def metric(logit, truth):
    prob = F.softmax(logit, 1)
    value, top = prob.topk(1, dim=1, largest=True, sorted=True)
    correct = top.eq(truth.view(-1, 1).expand_as(top))

    correct = correct.data.cpu().numpy()
    correct = np.mean(correct)
    return correct, prob

def do_valid( net, test_loader, criterion ):
    valid_num  = 0
    losses   = []
    corrects = []
    probs = []
    labels = []

    for input, truth in test_loader:
        b,n,c,w,h = input.size()
        input = input.view(b*n,c,w,h)

        input = input.cuda()
        truth = truth.cuda()

        with torch.no_grad():
            logit,_,_   = net(input)
            logit = logit.view(b,n,2)
            logit = torch.mean(logit, dim = 1, keepdim = False)

            truth = truth.view(logit.shape[0])
            loss    = criterion(logit, truth, False)
            correct, prob = metric(logit, truth)

        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())
        corrects.append(np.asarray(correct).reshape([1]))
        probs.append(prob.data.cpu().numpy())
        labels.append(truth.data.cpu().numpy())

    # assert(valid_num == len(test_loader.sampler))
    #----------------------------------------------

    correct = np.concatenate(corrects)
    loss    = np.concatenate(losses)
    loss    = loss.mean()
    correct = np.mean(correct)

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    tpr, fpr, acc = calculate_accuracy(0.5, probs[:,1], labels)
    acer,_,_,_,_ = ACER(0.5, probs[:, 1], labels)

    valid_loss = np.array([
        loss, acer, acc, correct
    ])

    return valid_loss,[probs[:, 1], labels]

def do_valid_test( net, test_loader, criterion ):
    valid_num  = 0
    losses   = []
    corrects = []
    probs = []
    labels = []


    for i, (input, truth) in enumerate(tqdm(test_loader)):
    # for input, truth in test_loader:
        b,n,c,w,h = input.size()
        input = input.view(b*n,c,w,h)

        input = input.cuda()
        truth = truth.cuda()

        with torch.no_grad():
            logit,_,_   = net(input)
            logit = logit.view(b,n,2)
            logit = torch.mean(logit, dim = 1, keepdim = False)

            truth = truth.view(logit.shape[0])
            loss    = criterion(logit, truth, False)
            correct, prob = metric(logit, truth)

        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())
        corrects.append(np.asarray(correct).reshape([1]))
        probs.append(prob.data.cpu().numpy())
        labels.append(truth.data.cpu().numpy())

    # assert(valid_num == len(test_loader.sampler))
    #----------------------------------------------

    correct = np.concatenate(corrects)
    loss    = np.concatenate(losses)
    loss    = loss.mean()
    correct = np.mean(correct)

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    tpr, fpr, acc = calculate_accuracy(0.5, probs[:,1], labels)
    acer,_,_,_,_ = ACER(0.5, probs[:, 1], labels)

    valid_loss = np.array([
        loss, acer, acc, correct
    ])

    return valid_loss,[probs[:, 1], labels]

def infer_test( net, test_loader):
    valid_num  = 0
    probs = []

    for i, (input, truth) in enumerate(tqdm(test_loader)):
        b,n,c,w,h = input.size()
        input = input.view(b*n,c,w,h)
        input = input.cuda()

        with torch.no_grad():
            logit,_,_   = net(input)
            logit = logit.view(b,n,2)
            logit = torch.mean(logit, dim = 1, keepdim = False)
            prob = F.softmax(logit, 1)

        valid_num += len(input)
        probs.append(prob.data.cpu().numpy())

    probs = np.concatenate(probs)
    return probs[:, 1]




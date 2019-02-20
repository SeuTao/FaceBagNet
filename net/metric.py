from include import *


def iou_accuracy(prob, truth, threshold=0.5,  is_average=True):
    batch_size = len(prob)
    p = prob.detach().view(batch_size,-1)
    t = truth.detach().view(batch_size,-1)

    p = p>threshold
    t = t>0.5
    intersection = p & t
    union        = p | t
    dice = (intersection.float().sum(1)+EPS) / (union.float().sum(1)+EPS)

    if is_average:
        dice = dice.sum()/batch_size
        return dice
    else:
        return dice



def pixel_accuracy(prob, truth, threshold=0.5,  is_average=True):
    batch_size = len(prob)
    p = prob.detach().view(batch_size,-1)
    t = truth.detach().view(batch_size,-1)

    p = p>threshold
    t = t>0.5
    correct = ( p == t).float()
    accuracy = correct.sum(1)/p.size(1)

    if is_average:
        accuracy = accuracy.sum()/batch_size
        return accuracy
    else:
        return accuracy





def accuracy(prob, truth, threshold=0.5,  is_average=True):
    batch_size = len(prob)
    p = prob.detach().view(-1)
    t = truth.detach().view(-1)
    p = p>threshold
    t = t>0.5
    correct  = ( p == t).float()
    accuracy = correct

    if is_average:
        accuracy = accuracy.sum()/batch_size
        return accuracy
    else:
        return accuracy




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
 
    print('\nsucess!')

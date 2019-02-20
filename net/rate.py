# learning rate schduler
# from include import *
import numpy as np
import math
import os

# http://elgoacademy.org/anatomy-matplotlib-part-1/
def plot_rates(fig, lrs, title=''):

    N = len(lrs)
    epoches = np.arange(0,N)


    #get limits
    max_lr  = np.max(lrs)
    xmin=0
    xmax=N
    dx=2

    ymin=0
    ymax=max_lr*1.2
    dy=(ymax-ymin)/10
    dy=10**math.ceil(math.log10(dy))

    ax = fig.add_subplot(111)
    #ax = fig.gca()
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.set_xticks(np.arange(xmin,xmax+0.0001, dx))
    ax.set_yticks(np.arange(ymin,ymax+0.0001, dy))
    ax.set_xlim(xmin,xmax+0.0001)
    ax.set_ylim(ymin,ymax+0.0001)
    ax.grid(b=True, which='minor', color='black', alpha=0.1, linestyle='dashed')
    ax.grid(b=True, which='major', color='black', alpha=0.4, linestyle='dashed')

    ax.set_xlabel('iter')
    ax.set_ylabel('learning rate')
    ax.set_title(title)
    ax.plot(epoches, lrs)



## simple stepping rates
class StepScheduler():
    def __init__(self, pairs):
        super(StepScheduler, self).__init__()

        N=len(pairs)
        rates=[]
        steps=[]
        for n in range(N):
            steps.append(pairs[n][0])
            rates.append(pairs[n][1])

        self.rates = rates
        self.steps = steps

    def __call__(self, epoch):

        N = len(self.steps)
        lr = -1
        for n in range(N):
            if epoch >= self.steps[n]:
                lr = self.rates[n]
        return lr

    def __str__(self):
        string = 'Step Learning Rates\n' \
                + 'rates=' + str(['%7.4f' % i for i in self.rates]) + '\n' \
                + 'steps=' + str(['%7.0f' % i for i in self.steps]) + ''
        return string


## https://github.com/pytorch/tutorials/blob/master/beginner_source/transfer_learning_tutorial.py
class DecayScheduler():
    def __init__(self, base_lr, decay, step):
        # super(DecayScheduler, self).__init__()
        self.step  = step
        self.decay = decay
        self.base_lr = base_lr

    def get_rate(self, epoch):
        lr = self.base_lr * (self.decay**(epoch // self.step))
        return lr



    def __str__(self):
        string = '(Exp) Decay Learning Rates\n' \
                + 'base_lr=%0.3f, decay=%0.3f, step=%0.3f'%(self.base_lr, self.decay, self.step)
        return string




# 'Cyclical Learning Rates for Training Neural Networks'- Leslie N. Smith, arxiv 2017
#       https://arxiv.org/abs/1506.01186
#       https://github.com/bckenstler/CLR

class CyclicScheduler0():

    def __init__(self, min_lr=0.001, max_lr=0.01, period=10 ):
        super(CyclicScheduler, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.period = period

    def __call__(self, time):

        #sawtooth
        #r = (1-(time%self.period)/self.period)

        #cosine
        time= time%self.period
        r = (np.cos(time/self.period *PI)+1)/2

        lr = self.min_lr + r*(self.max_lr-self.min_lr)
        return lr

    def __str__(self):
        string = 'CyclicScheduler\n' \
                + 'min_lr=%0.3f, max_lr=%0.3f, period=%8.1f'%(self.min_lr, self.max_lr, self.period)
        return string


class CyclicScheduler1():

    def __init__(self, min_lr=0.001, max_lr=0.01, period=10, max_decay=0.99, warm_start=0 ):
        super(CyclicScheduler, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.period = period
        self.max_decay = max_decay
        self.warm_start = warm_start
        self.cycle = -1

    def __call__(self, time):
        if time<self.warm_start: return self.max_lr

        #cosine
        self.cycle = (time-self.warm_start)//self.period
        time = (time-self.warm_start)%self.period

        period = self.period
        min_lr = self.min_lr
        max_lr = self.max_lr *(self.max_decay**self.cycle)


        r   = (np.cos(time/period *PI)+1)/2
        lr = min_lr + r*(max_lr-min_lr)

        return lr



    def __str__(self):
        string = 'CyclicScheduler\n' \
                + 'min_lr=%0.4f, max_lr=%0.4f, period=%8.1f'%(self.min_lr, self.max_lr, self.period)
        return string


#tanh curve
class CyclicScheduler():

    def __init__(self, min_lr=0.001, max_lr=0.01, period=10, max_decay=0.99, warm_start=0 ):
        super(CyclicScheduler, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.period = period
        self.max_decay = max_decay
        self.warm_start = warm_start
        self.cycle = -1

    def __call__(self, time):
        if time<self.warm_start: return self.max_lr

        #cosine
        self.cycle = (time-self.warm_start)//self.period
        time = (time-self.warm_start)%self.period

        period = self.period
        min_lr = self.min_lr
        max_lr = self.max_lr *(self.max_decay**self.cycle)


        r   = (np.tanh(-time/period *16 +8)+1)*0.5
        lr = min_lr + r*(max_lr-min_lr)

        return lr



    def __str__(self):
        string = 'CyclicScheduler\n' \
                + 'min_lr=%0.3f, max_lr=%0.3f, period=%8.1f'%(self.min_lr, self.max_lr, self.period)
        return string

#
# class CyclicScheduler():
#
#     def __init__(self, pairs, period=10, max_decay=1, warm_start=0 ):
#         super(CyclicScheduler, self).__init__()
#
#         self.lrs=[]
#         self.steps=[]
#         for p in pairs:
#             self.steps.append(p[0])
#             self.lrs.append(p[1])
#
#
#         self.period = period
#         self.warm_start = warm_start
#         self.max_decay = max_decay
#         self.cycle = -1
#
#     def __call__(self, time):
#         if time<self.warm_start: return self.lrs[0]
#
#         self.cycle = (time-self.warm_start)//self.period
#         time = (time-self.warm_start)%self.period
#
#         rates = self.lrs.copy()
#         steps = self.steps
#         rates[0] = rates[0] *(self.max_decay**self.cycle)
#         lr = -1
#         for rate,step in zip(rates,steps):
#             if time >= step:
#                lr = rate
#
#         return lr
#
#
#
#     def __str__(self):
#         string = 'CyclicScheduler\n' \
#                 + 'lrs  =' + str(['%7.4f' % i for i in self.lrs]) + '\n' \
#                 + 'steps=' + str(['%7.0f' % i for i in self.steps]) + '\n' \
#                 + 'period=%8.1f'%(self.period)
#         return string


class NullScheduler():
    def __init__(self, lr=0.01 ):
        super(NullScheduler, self).__init__()
        self.lr    = lr
        self.cycle = 0

    def __call__(self, time):
        return self.lr

    def __str__(self):
        string = 'NullScheduler\n' \
                + 'lr=%0.5f '%(self.lr)
        return string


# net ------------------------------------
# https://github.com/pytorch/examples/blob/master/imagenet/main.py ###############
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    num_iters=125


    #scheduler = StepScheduler([ (0,0.1),  (10,0.01),  (25,0.005),  (35,0.001), (40,0.0001), (43,-1)])
    #scheduler = DecayScheduler(base_lr=0.1, decay=0.32, step=10)
    scheduler = CyclicScheduler(min_lr=0.0001, max_lr=0.01, period=30., warm_start=5) ##exp_range ##triangular2
    #scheduler = CyclicScheduler([ (0,0.1),  (25,0.01),  (45,0.005)], period=50., warm_start=5) ##exp_range ##triangular2


    lrs = np.zeros((num_iters),np.float32)
    for iter in range(num_iters):

        lr = scheduler(iter)
        lrs[iter] = lr
        if lr<0:
            num_iters = iter
            break
        print ('iter=%02d,  lr=%f   %d'%(iter,lr, scheduler.cycle))


    #plot
    fig = plt.figure()
    plot_rates(fig, lrs, title=str(scheduler))
    plt.show()


#  https://github.com/Jiaming-Liu/pytorch-lr-scheduler/blob/master/lr_scheduler.py
#  PVANET plateau lr policy
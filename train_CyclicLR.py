import os
os.environ['CUDA_VISIBLE_DEVICES'] =  '5' #'3,2,1,0'
import sys
sys.path.append("..")
import argparse
from process.data import *
from process.augmentation import *
from net.rate import *
from file import *
from metric import *
from loss.cyclic_lr import CosineAnnealingLR_with_Restart

def get_model(model_name, num_class,is_first_bn):
    if model_name == 'resnet18':
        from model.model_resnet18 import Net
    elif model_name == 'seresnext18':
        from model.model_seresnext18 import Net
    elif model_name == 'resnet34':
        from model.model_resnet34 import Net
    elif model_name == 'bagnet33':
        from model.model_bagnet33 import Net

    net = Net(num_class=num_class,is_first_bn=is_first_bn)
    return net

def get_augment(image_mode):
    if image_mode == 'color':
        augment = color_augumentor
    elif image_mode == 'depth':
        augment = depth_augumentor
    elif image_mode == 'ir':
        augment = ir_augumentor
    return augment

def run_train(config):
    out_dir = './models'
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoint = config.pretrained_model

    # schduler = DecayScheduler(base_lr=0.1, decay=0.1, step=20) #it means 3 epoch for 0.01, 3 epoch for 0.001
    criterion  = softmax_cross_entropy_criterion

    ## setup  -----------------------------------------------------------------------------
    if not os.path.exists(out_dir +'/checkpoint'):
        os.makedirs(out_dir +'/checkpoint')
    if not os.path.exists(out_dir +'/backup'):
        os.makedirs(out_dir +'/backup')
    if not os.path.exists(out_dir +'/backup'):
        os.makedirs(out_dir +'/backup')

    log = Logger()
    log.open(os.path.join(out_dir,config.model_name+'.txt'),mode='a')
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... xxx baseline  ... \n')
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    augment = get_augment(config.image_mode)

    train_dataset = FDDataset(mode = 'train', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index,augment=augment)
    train_loader  = DataLoader(train_dataset,
                                shuffle=True,
                                batch_size  = config.batch_size,
                                drop_last   = True,
                                num_workers = 4)

    valid_dataset = FDDataset(mode = 'val', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index,augment=augment)
    valid_loader  = DataLoader( valid_dataset,
                                shuffle=False,
                                batch_size = config.batch_size // 10,
                                drop_last  = False,
                                num_workers = 4)

    assert(len(train_dataset)>=config.batch_size)
    log.write('batch_size = %d\n'%(config.batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')
    log.write('** net setting **\n')

    net = get_model(model_name=config.model, num_class=2, is_first_bn=True)
    print(net)
    net = torch.nn.DataParallel(net)
    net =  net.cuda()

    if initial_checkpoint is not None:
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n'%(type(net)))
    log.write('criterion=%s\n'%criterion)
    log.write('\n')

    iter_smooth = 20
    start_iter = 0

    # log.write('schduler\n  %s\n'%(schduler))
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('                    |------------ VALID -------------|-------- TRAIN/BATCH ----------|         \n')
    log.write('rate   iter  epoch  | loss   acc-1  acc-3   lb       | loss   acc-1  acc-3   lb      |  time   \n')
    log.write('----------------------------------------------------------------------------------------------------\n')

    iter = 0
    i    = 0

    train_loss = np.zeros(6, np.float32)
    valid_loss = np.zeros(6, np.float32)
    batch_loss = np.zeros(6, np.float32)

    start = timer()
    #-----------------------------------------------
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.1, momentum=0.9, weight_decay=0.0005)

    sgdr = CosineAnnealingLR_with_Restart(optimizer,
                                          T_max=config.cycle_inter,
                                          T_mult=1,
                                          model=net,
                                          out_dir='../input/',
                                          take_snapshot=False,
                                          eta_min=1e-3)

    # net.eval()
    # valid_loss, _ = do_valid_test(net, valid_loader, criterion)
    # net.train()

    global_min_acer = 1.0
    for cycle_index in range(config.cycle_num):
        print('cycle index: ' + str(cycle_index))
        min_acer = 1.0

        for epoch in range(0, config.cycle_inter):
            sgdr.step()
            lr = optimizer.param_groups[0]['lr']
            print('lr : {:.4f}'.format(lr))

            # for epoch in range(train_epoch):
            sum_train_loss = np.zeros(6,np.float32)
            sum = 0
            optimizer.zero_grad()

            for input, truth in train_loader:
                iter = i + start_iter

                # one iteration update  -------------
                net.train()
                input = input.cuda()
                truth = truth.cuda()

                logit,_,_ = net.forward(input)
                truth = truth.view(logit.shape[0])

                loss  = criterion(logit, truth)
                precision,_ = metric(logit, truth)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # print statistics  ------------
                batch_loss[:2] = np.array(( loss.item(), precision.item(),))

                sum += 1
                if iter%iter_smooth == 0:
                    train_loss = sum_train_loss/sum
                    sum = 0
                i=i+1


            if epoch >= config.cycle_inter // 2:
                net.eval()
                valid_loss,_ = do_valid_test(net, valid_loader, criterion)
                net.train()

                if valid_loss[1] < min_acer and epoch > 0:
                    min_acer = valid_loss[1]
                    ckpt_name = out_dir + '/checkpoint/Cycle_' + str(cycle_index) + '_min_acer_model.pth'
                    torch.save(net.state_dict(), ckpt_name)
                    log.write('save cycle ' + str(cycle_index) + ' min acer model: ' + str(min_acer) + '\n')

                if valid_loss[1] < global_min_acer and epoch > 0:
                    global_min_acer = valid_loss[1]
                    ckpt_name = out_dir + '/checkpoint/global_min_acer_model.pth'
                    torch.save(net.state_dict(), ckpt_name)
                    log.write('save global min acer model: ' + str(min_acer) + '\n')

            asterisk = ' '
            log.write(config.model_name+'Cycle: %5.1f %0.4f %5.1f %6.1f | %0.6f  %0.6f  %0.3f  (%0.3f)%s  | %0.6f  %0.6f  %0.3f  (%0.3f) |%s \n' % (
                cycle_index, lr, iter, epoch,
                valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3],asterisk,
                batch_loss[0], 0.0, 0.0, batch_loss[1],
                time_to_str((timer() - start), 'min')))

            print(config.model_name+'Cycle: %5.1f %0.4f %5.1f %6.1f | %0.6f  %0.6f  %0.3f  (%0.3f)%s  | %0.6f  %0.6f  %0.3f  (%0.3f)  | %s' % (
                cycle_index, lr, iter, epoch,
                valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3],' ',
                batch_loss[0], 0.0, 0.0, batch_loss[1],
                time_to_str((timer() - start),'min')))

        ckpt_name = out_dir + '/checkpoint/Cycle_' + str(cycle_index) + '_final_model.pth'
        torch.save(net.state_dict(), ckpt_name)
        log.write('save cycle ' + str(cycle_index) + ' final model \n')


# def run_valid_10crop(config):
#     out_dir = './models'
#     out_dir = os.path.join(out_dir,config.model_name)
#     initial_checkpoint = config.pretrained_model
#
#     augment = get_augment(config.image_mode)
#
#     ## net ---------------------------------------
#     net = get_model(model_name=config.model, num_class=2,is_first_bn=True)
#     print(net)
#     net = torch.nn.DataParallel(net)
#     net =  net.cuda()
#
#     if initial_checkpoint is not None:
#         initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
#         print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
#         net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
#
#     valid_dataset = FDDataset(mode = 'val',
#                               modality=config.image_mode,
#                               image_size=config.image_size,
#                               fold_index=config.train_fold_index,
#                               augment=augment)
#
#     valid_loader  = DataLoader( valid_dataset,
#                                 shuffle=False,
#                                 batch_size  = config.batch_size // 10,
#                                 drop_last   = False,
#                                 num_workers=8)
#
#     criterion = softmax_cross_entropy_criterion
#     net.eval()
#     valid_loss,out = do_valid_test(net, valid_loader, criterion)
#     net.train()
#     print('%0.6f  %0.6f  %0.3f  (%0.3f) \n' % (valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3]))

def run_valid_10crop(config, dir):
    out_dir = './models'
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoint = config.pretrained_model
    augment = get_augment(config.image_mode)

    ## net ---------------------------------------
    net = get_model(model_name=config.model, num_class=2, is_first_bn=True)
    net = torch.nn.DataParallel(net)
    net =  net.cuda()

    if initial_checkpoint is not None:
        save_dir = os.path.join(out_dir + '/checkpoint', dir, initial_checkpoint)
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        if not os.path.exists(os.path.join(out_dir + '/checkpoint', dir)):
            os.makedirs(os.path.join(out_dir + '/checkpoint', dir))


    valid_dataset = FDDataset(mode = 'val', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index,augment=augment)
    valid_loader  = DataLoader( valid_dataset,
                                shuffle=False,
                                batch_size  = config.batch_size,
                                drop_last   = False,
                                num_workers=8)

    criterion = softmax_cross_entropy_criterion
    net.eval()
    valid_loss,out = do_valid_test(net, valid_loader, criterion)
    net.train()
    print('%0.6f  %0.6f  %0.3f  (%0.3f) \n' % (valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3]))

    tpr, fpr, acc = calculate_accuracy(0.5, out[0], out[1])
    acer, tp, fp, tn, fn = ACER(0.5, out[0], out[1])

    print('ensemble')
    print(acer)
    print(acc)
    print('tp: '+str(tp))
    print('tn: '+str(tn))
    print('fp: '+str(fp))
    print('fn: '+str(fn))


    submission(out[0],save_dir+'_noTTA.txt')

    # acer_min = 1.0
    # thres_min = 0.0
    # re = []
    #
    # for thres in np.arange(0.0, 1.0, 0.01):
    #     acer, tp, fp, tn, fn= ACER(thres, out[0], out[1])
    #     if acer < acer_min:
    #         acer_min = acer
    #         thres_min = thres
    #         re = [tp, fp, tn, fn]
    #
    # print('\n ajust threshold')
    # print(acer_min)
    # print(thres_min)
    #
    # tp, fp, tn, fn = re
    # print('tp: '+str(tp))
    # print('tn: '+str(tn))
    # print('fp: '+str(fp))
    # print('fn: '+str(fn))

    try:
        TPR_FPR(out[0], out[1], fpr_target = 0.01)
        TPR_FPR(out[0], out[1], fpr_target = 0.001)
        TPR_FPR(out[0], out[1], fpr_target = 0.0001)
    except:
        return

def run_test_10crop(config, dir):
    out_dir = './models'
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoint = config.pretrained_model
    augment = get_augment(config.image_mode)

    ## net ---------------------------------------
    net = get_model(model_name=config.model, num_class=2, is_first_bn=True)
    net = torch.nn.DataParallel(net)
    net =  net.cuda()

    if initial_checkpoint is not None:
        save_dir = os.path.join(out_dir + '/checkpoint', dir, initial_checkpoint)
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        if not os.path.exists(os.path.join(out_dir + '/checkpoint', dir)):
            os.makedirs(os.path.join(out_dir + '/checkpoint', dir))


    valid_dataset = FDDataset(mode = 'val', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index,augment=augment)
    valid_loader  = DataLoader( valid_dataset,
                                shuffle=False,
                                batch_size  = config.batch_size,
                                drop_last   = False,
                                num_workers=8)

    test_dataset = FDDataset(mode = 'test', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index,augment=augment)
    test_loader  = DataLoader( test_dataset,
                                shuffle=False,
                                batch_size  = config.batch_size,
                                drop_last   = False,
                                num_workers=8)

    criterion = softmax_cross_entropy_criterion
    net.eval()

    valid_loss,out = do_valid_test(net, valid_loader, criterion)
    print('%0.6f  %0.6f  %0.3f  (%0.3f) \n' % (valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3]))

    print('infer!!!!!!!!!')
    out = infer_test(net, test_loader)
    print('done')

    submission(out,save_dir+'_noTTA.txt')

def main(config):

    if config.mode == 'train':
        run_train(config)

    if config.mode == 'test':
        run_train(config)

    if config.mode == 'valid_10crop':
        run_valid_10crop(config)

    if config.mode == 'inferAllcycle_val':
        for i in range(config.cycle_num):
            config.pretrained_model = r'Cycle_'+str(i)+'_final_model.pth'
            run_valid_10crop(config, dir = 'final')

        for i in range(config.cycle_num):
            config.pretrained_model = r'Cycle_'+str(i)+'_min_acer_model.pth'
            run_valid_10crop(config, dir = 'min')

        config.pretrained_model = r'global_min_acer_model.pth'
        run_valid_10crop(config, dir='global')

    if config.mode == 'inferAllcycle_test':
    #     for i in range(config.cycle_num):
    #         config.pretrained_model = r'Cycle_'+str(i)+'_final_model.pth'
    #         run_test_10crop(config, dir = 'final_test')

    #     for i in range(config.cycle_num):
    #         config.pretrained_model = r'Cycle_'+str(i)+'_min_acer_model.pth'
    #         run_test_10crop(config, dir = 'min_test')

        config.pretrained_model = r'global_min_acer_model.pth'
        run_test_10crop(config, dir='global_test')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default = -1)
    parser.add_argument('--model', type=str, default='seresnext18')

    parser.add_argument('--image_mode', type=str, default='ir')
    parser.add_argument('--model_name', type=str, default='seresnext18_NewCrop_ir_with_pretrain_112to48_SnapshotEnsemble_NewVal')

    parser.add_argument('--image_size', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--cycle_num', type=int, default=10)
    parser.add_argument('--cycle_inter', type=int, default=50)

    # parser.add_argument('--mode', type=str, default='train', choices=['train','test','valid_10crop'])
    # parser.add_argument('--pretrained_model', type=str, default=None)

    parser.add_argument('--mode', type=str, default='inferAllcycle_test', choices=['train','test','valid_10crop'])
    parser.add_argument('--pretrained_model', type=str, default=None)

    parser.add_argument('--iter_save_interval', type=int, default= 1000 * 2)
    parser.add_argument('--iter_valid', type=int, default=100)
    parser.add_argument('--num_iters', type=int, default=200 * 1000 *2)
    parser.add_argument('--step', type=int, default=50 * 1000)

    config = parser.parse_args()
    print(config)
    main(config)
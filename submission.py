from metric import *
from process.data_fusion import *

DATA_ROOT = r'/data1/shentao/DATA/CVPR19_FaceAntiSpoofing'

def load_sub(sub = r'r18_val0.458.txt'):
    sub_dict = {}
    f = open(sub,'r')

    lines = f.readlines()

    for line in lines:
        line = line.strip()
        line = line.split(' ')
        sub_dict[line[0]] = float(line[3])

    return sub_dict

def ensemble(sub_list, save_name):
    dict_list = []
    for sub in sub_list:
        sub_dict = load_sub(sub)
        dict_list.append(sub_dict)

    test_list = load_val_list()
    probs = []
    labels = []
    for name,_,_,label in test_list:

        prob_tmp = 0.0
        for sub_dict in dict_list:
            prob_tmp += sub_dict[name] / (len(dict_list)*1.0)

        probs.append(prob_tmp)
        labels.append(int(label))

    probs = np.asarray(probs)
    labels = np.asarray(labels)

    # tpr, fpr, acc = calculate_accuracy(0.5, probs, labels)
    acer, tp, fp, tn, fn = ACER(0.5, probs, labels)
    print('\nonline')
    print(acer)
    print('tp: '+str(tp))
    print('tn: '+str(tn))
    print('fp: '+str(fp))
    print('fn: '+str(fn))

    submission(probs,save_name)
    try:
        TPR_FPR(probs, labels, fpr_target = 0.01)
        TPR_FPR(probs, labels, fpr_target = 0.001)
        TPR_FPR(probs, labels, fpr_target = 0.0001)
    except:
        return

def ensemble_valid_dir(sub_dir_list, save_name):
    dict_list = []
    for sub_dir in sub_dir_list:
        print(len(os.listdir(sub_dir)))
        for sub in os.listdir(sub_dir):
            if '.txt' in sub:
                sub_dict = load_sub(os.path.join(sub_dir,sub))
                dict_list.append(sub_dict)

    test_list = load_val_list()

    probs = []
    labels = []
    for name,_,_,label in test_list:

        prob_tmp = 0.0
        for sub_dict in dict_list:
            prob_tmp += sub_dict[name] / (len(dict_list)*1.0)

        probs.append(prob_tmp)
        labels.append(int(label))

    probs = np.asarray(probs)
    labels = np.asarray(labels)

    acer, tp, fp, tn, fn = ACER(0.5, probs, labels)
    print('\nonline')
    print(acer)
    print('tp: '+str(tp))
    print('tn: '+str(tn))
    print('fp: '+str(fp))
    print('fn: '+str(fn))

    acer_min = 1.0
    thres_min = 0.0
    re = []
    for thres in np.arange(0.0, 1.0, 0.01):
        acer, tp, fp, tn, fn= ACER(thres, probs, labels)
        if acer < acer_min:
            acer_min = acer
            thres_min = thres
            re = [tp, fp, tn, fn]

    print('ajust')
    print(acer_min)
    print(thres_min)

    tp, fp, tn, fn = re
    print('tp: '+str(tp))
    print('tn: '+str(tn))
    print('fp: '+str(fp))
    print('fn: '+str(fn))

    submission(probs,save_name, mode='valid')
    try:
        TPR_FPR(probs, labels, fpr_target = 0.01)
        TPR_FPR(probs, labels, fpr_target = 0.001)
        TPR_FPR(probs, labels, fpr_target = 0.0001)
    except:
        return

def ensemble_test_dir(sub_dir_list, save_name):
    dict_list = []
    for sub_dir in sub_dir_list:
        print(len(os.listdir(sub_dir)))
        for sub in os.listdir(sub_dir):
            if '.txt' in sub:
                sub_dict = load_sub(os.path.join(sub_dir,sub))
                dict_list.append(sub_dict)
    test_list = load_test_list()

    probs = []
    for name,_,_ in test_list:
        prob_tmp = 0.0
        for sub_dict in dict_list:
            prob_tmp += sub_dict[name] / (len(dict_list)*1.0)
        probs.append(prob_tmp)

    probs = np.asarray(probs)
    submission(probs,save_name, mode='test')

def compare(sub=r'r18_val.txt'):
    sub_dict = load_sub(sub)
    print(len(sub_dict))
    test_list = load_val_list()

    probs = []
    labels = []
    for name,_,_,label in test_list:
        probs.append(sub_dict[name])
        labels.append(int(label))

    probs = np.asarray(probs)
    labels = np.asarray(labels)

    # acer_min = 1.0
    # thres_min = 0.0
    # re = []
    # for thres in np.arange(0.0, 1.0, 0.005):
    #
    #     acer,tp, fp, tn, fn = ACER(thres, probs, labels)
    #     if acer < acer_min:
    #         acer_min = acer
    #         thres_min = thres
    #         re = [tp, fp, tn, fn]
    #
    # print(acer_min)
    # print(thres_min)
    #
    # tp, fp, tn, fn = re
    #
    # print('tp: '+str(tp))
    # print('tn: '+str(tn))
    # print('fp: '+str(fp))
    # print('fn: '+str(fn))

    acer, tp, fp, tn, fn = ACER(0.5, probs, labels)
    print('\nonline')
    print('acer: ' + str(acer))
    print('tp: '+str(tp))
    print('tn: '+str(tn))
    print('fp: '+str(fp))
    print('fn: '+str(fn))

    # TPR_FPR( probs, labels)
    try:
        TPR_FPR(probs, labels, fpr_target = 0.01)
        TPR_FPR(probs, labels, fpr_target = 0.001)
        TPR_FPR(probs, labels, fpr_target = 0.0001)
    except:
        return

    #TP: 2987 TN:6533 FP:81 FN:7
    # test_pos_num:2688
    # thres: 0.14
    # 0.04252210638590696
    # tp: 2614 gt: 2987
    # tn: 6522 gt: 6533
    # fp: 398  gt: 81
    # fn: 74   gt: 7

def extract(sub, dir):
    thres = 0.5

    sub_dict = load_sub(sub)
    test_list = load_test_list()

    # fn
    fp_list = []
    fn_list = []

    for c,d,i,label in test_list:
        prob_tmp = sub_dict[c]
        label_tmp = int(label)

        # fp
        if prob_tmp >= thres and label_tmp==0:
            fp_list.append([c,d,i])

        # fn
        if prob_tmp < thres and label_tmp==1:
            fn_list.append([c,d,i])

        # break
    print(len(fp_list))
    print(len(fn_list))
    import shutil

    save_dir = os.path.join(dir,'fp')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for c,d,i in fp_list:
        c = os.path.join(DATA_ROOT, c)
        d = os.path.join(DATA_ROOT, d)
        i = os.path.join(DATA_ROOT, i)

        c_new = os.path.join(save_dir,os.path.split(c)[1])
        d_new = os.path.join(save_dir,os.path.split(d)[1])
        i_new = os.path.join(save_dir,os.path.split(i)[1])

        shutil.copyfile(c, c_new)
        shutil.copyfile(d, d_new)
        shutil.copyfile(i, i_new)

    save_dir = os.path.join(dir,'fn')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for c, d, i in fn_list:
        c = os.path.join(DATA_ROOT, c)
        d = os.path.join(DATA_ROOT, d)
        i = os.path.join(DATA_ROOT, i)

        c_new = os.path.join(save_dir, os.path.split(c)[1])
        d_new = os.path.join(save_dir, os.path.split(d)[1])
        i_new = os.path.join(save_dir, os.path.split(i)[1])

        shutil.copyfile(c, c_new)
        shutil.copyfile(d, d_new)
        shutil.copyfile(i, i_new)

    return

def extract_imgs(sub, dir):

    thres = 0.95
    thres_up = 0.98

    print(thres)
    print(thres_up)

    sub_dict = load_sub(sub)
    test_list = load_test_list()

    # fn
    fp_list = []

    for c,d,i in test_list:
        prob_tmp = sub_dict[c]

        if prob_tmp >= thres and prob_tmp < thres_up:
            fp_list.append([c,d,i])

    print(len(fp_list))

    import shutil
    save_dir = dir
    if not os.path.exists(dir):
        os.makedirs(dir)

    print(dir)

    for c,d,i in fp_list:
        c = os.path.join(DATA_ROOT, c)
        d = os.path.join(DATA_ROOT, d)
        i = os.path.join(DATA_ROOT, i)

        c_new = os.path.join(save_dir,os.path.split(c)[1])
        d_new = os.path.join(save_dir,os.path.split(d)[1])
        i_new = os.path.join(save_dir,os.path.split(i)[1])

        shutil.copyfile(c, c_new)
        shutil.copyfile(d, d_new)
        shutil.copyfile(i, i_new)

def compare_hand_label():

    dir_fn_n = r'./0110_out_0/fn_n'
    dir_fp_p = r'./0110_out_0/fp_p'

    def get_color_dict(dir):
        list = os.listdir(dir)

        dict={}
        for tmp in list:
            if 'color' in tmp:
                dict[tmp] = 1

        return dict

    dict_fn_n = get_color_dict(dir_fn_n)
    dict_fp_p = get_color_dict(dir_fp_p)

    # to be removed
    print(len(dict_fn_n))

    # to be added
    print(len(dict_fp_p))

    current_dir = r'/data1/shentao/DATA/CVPR19_FaceAntiSpoofing/Val_label/20190110_val_real'

    for item in dict_fp_p:
        path_tmp = os.path.join(current_dir,item)
        if not os.path.exists(path_tmp):
            img_tmp = np.zeros([10,10]).astype(np.uint8)
            cv2.imwrite(path_tmp,img_tmp)

            print(path_tmp)
            print('add!!!!!!!!')

    for item in dict_fn_n:
        path_tmp = os.path.join(current_dir,item)
        if os.path.exists(path_tmp):
            os.remove(path_tmp)
            print(path_tmp)
            print('remove!!!!!!!!')

def find_thres(sub=r'r18_val.txt', pos_num = 2994+909 ):
    f = open(sub,'r')
    lines = f.readlines()
    f.close()

    list = []
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        prob = float(line[3])
        list.append(prob)

    probs = np.asarray(list)

    acer_min = 1.0
    thres_min = 0.0

    for thres in np.arange(0.0, 1.0, 0.001):
        num_ = probs>thres
        num_ = np.sum(num_)

        print(num_)
        if num_ <= pos_num:
            print(thres)
            break

def nrom_thres(sub = r'r18_val.txt', thres = 0.855):

    f = open(sub,'r')
    lines = f.readlines()
    f.close()

    f = open(sub+'thres_change.txt','w')

    for line in lines:
        line_ = line.strip()
        line_ = line_.split(' ')

        prob = float(line_[3])

        if prob<thres:
            prob_nrom = prob / thres * 0.5
        else:
            prob_nrom = (prob-thres) / (1.0-thres) * 0.5 + 0.5

        f.write(line_[0]+' '+line_[1]+' '+line_[2]+' '+str(prob_nrom)+'\n')

    f.close()
    return

def nrom_thres_(sub = r'r18_val.txt', thres = 0.5):

    f = open(sub,'r')
    lines = f.readlines()
    f.close()

    f = open('r18_fusion_4tAve_9cropTTA.txt','r')
    lines_ = f.readlines()
    f.close()

    f = open('tmp.txt','w')
    for line, line2 in zip(lines,lines_):
        line_ = line.strip()
        line_ = line_.split(' ')
        # prob = float(line_[3])

        line2_ = line2.strip()
        line2_ = line2_.split(' ')
        prob2 = float(line2_[3])

        if prob2<thres:
            prob_nrom = prob2 / thres * 0.4 + 0.1
        else:
            prob_nrom = (prob2-thres) / (1.0-thres) * 0.4 + 0.5

        f.write(line_[0]+' '+line_[1]+' '+line_[2]+' '+str(prob_nrom)+'\n')

    f.close()
    return

def sub_former():
    sub_ir = [r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_ir_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_0_min_acer_model.pth0.0299_noTTA.txt',
                 r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_ir_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_1_min_acer_model.pth0.1650_noTTA.txt',
                 r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_ir_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_2_min_acer_model.pth0.0231_noTTA.txt',
                 r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_ir_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_3_min_acer_model.pth0.0235_noTTA.txt',
                 r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_ir_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_4_min_acer_model.pth0.0373_noTTA.txt',
                 r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_ir_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_5_min_acer_model.pth0.0330_noTTA.txt',
                 r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_ir_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_6_min_acer_model.pth0.0165_noTTA.txt',
                 r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_ir_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_7_min_acer_model.pth0.0400_noTTA.txt',
                 r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_ir_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_8_min_acer_model.pth0.0172_noTTA.txt',
                 r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_ir_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_9_min_acer_model.pth0.0350_noTTA.txt',
                 ]

    sub_color = [
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_color_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_0_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_color_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_1_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_color_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_2_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_color_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_3_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_color_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_4_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_color_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_5_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_color_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_6_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_color_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_7_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_color_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_8_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_color_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_9_min_acer_model.pth_noTTA.txt',
        ]

    sub_depth = [
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_depth_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_0_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_depth_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_1_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_depth_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_2_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_depth_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_3_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_depth_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_4_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_depth_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_5_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_depth_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_6_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_depth_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_7_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_depth_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_8_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_pretrain_48_depth_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_9_min_acer_model.pth_noTTA.txt',
        ]

    sub_fusion = [
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_fusion_pretrain48_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_0_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_fusion_pretrain48_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_1_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_fusion_pretrain48_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_2_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_fusion_pretrain48_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_3_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_fusion_pretrain48_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_4_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_fusion_pretrain48_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_5_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_fusion_pretrain48_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_6_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_fusion_pretrain48_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_7_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_fusion_pretrain48_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_8_min_acer_model.pth_noTTA.txt',
        r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/models/r18_fusion_pretrain48_fold-1_rotate_RC0.6_SnapshotEnsemble/checkpoint/Cycle_9_min_acer_model.pth_noTTA.txt',
        ]

def sub_first():
    dir = r'./models/'
    dir_list = [
                dir + r'r18_fusion_NewCrop_pretrain_112to32_SnapshotEnsemble_NewVal/checkpoint/global',
                dir + r'r18_fusion_NewCrop_pretrain_112to48_SnapshotEnsemble_NewVal/checkpoint/global',
                dir + r'r18_fusion_NewCrop_pretrain_112to64_SnapshotEnsemble_NewVal/checkpoint/global',

                dir + r'seresnext18_NewCrop_color_with_pretrain_112to48_SnapshotEnsemble_NewVal/checkpoint/global',
                dir + r'seresnext18_NewCrop_depth_with_pretrain_112to48_SnapshotEnsemble_NewVal/checkpoint/global',
                dir + r'seresnext18_NewCrop_ir_with_pretrain_112to48_SnapshotEnsemble_NewVal/checkpoint/global',
                ]

    ensemble_valid_dir(dir_list, 'valid_first.txt')

    dir_list = [
                dir + r'r18_fusion_NewCrop_pretrain_112to32_SnapshotEnsemble_NewVal/checkpoint/global_test',
                dir + r'r18_fusion_NewCrop_pretrain_112to48_SnapshotEnsemble_NewVal/checkpoint/global_test',
                dir + r'r18_fusion_NewCrop_pretrain_112to64_SnapshotEnsemble_NewVal/checkpoint/global_test',
                dir + r'seresnext18_NewCrop_color_with_pretrain_112to48_SnapshotEnsemble_NewVal/checkpoint/global_test',
                dir + r'seresnext18_NewCrop_depth_with_pretrain_112to48_SnapshotEnsemble_NewVal/checkpoint/global_test',
                dir + r'seresnext18_NewCrop_ir_with_pretrain_112to48_SnapshotEnsemble_NewVal/checkpoint/global_test'
                ]

    ensemble_test_dir(dir_list, 'test_first.txt')


if __name__ == '__main__':
    sub_first()


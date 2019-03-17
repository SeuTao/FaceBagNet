from metric import *
from process.data_fusion import *

def load_sub(sub):
    sub_dict = {}
    f = open(sub,'r')

    lines = f.readlines()

    for line in lines:
        line = line.strip()
        line = line.split(' ')
        sub_dict[line[0]] = float(line[3])

    return sub_dict

def ensemble_test_dir(sub_dir_list, save_name):
    dict_list = []
    for sub_dir in sub_dir_list:
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

def sub_first():
    dir = r'./models/'

    dir_list = [dir + r'baseline_fusion_32/checkpoint/global_test_36_TTA',
                dir + r'baseline_fusion_48/checkpoint/global_test_36_TTA',
                dir + r'baseline_fusion_64/checkpoint/global_test_36_TTA',

                dir + r'model_A_color_48/checkpoint/global_test_36_TTA',
                dir + r'model_A_depth_48/checkpoint/global_test_36_TTA',
                dir + r'model_A_ir_48/checkpoint/global_test_36_TTA']

    ensemble_test_dir(dir_list, 'test_first.txt')
    print('test_first.txt done!')

def sub_second():
    dir = r'./models/'

    dir_list = [dir + r'model_A_color_48/checkpoint/global_test_36_TTA',
                dir + r'model_A_depth_48/checkpoint/global_test_36_TTA',
                dir + r'model_A_ir_48/checkpoint/global_test_36_TTA',

                dir + r'model_A_color_48/checkpoint/global_test_36_TTA',
                dir + r'model_A_depth_48/checkpoint/global_test_36_TTA',
                dir + r'model_A_ir_48/checkpoint/global_test_36_TTA',

                dir + r'model_A_color_32/checkpoint/global_test_36_TTA',
                dir + r'model_A_depth_32/checkpoint/global_test_36_TTA',
                dir + r'model_A_ir_32/checkpoint/global_test_36_TTA',

                dir + r'model_A_color_64/checkpoint/global_test_36_TTA',
                dir + r'model_A_depth_64/checkpoint/global_test_36_TTA',
                dir + r'model_A_ir_64/checkpoint/global_test_36_TTA',]

    ensemble_test_dir(dir_list, 'test_second.txt')
    print('test_second.txt done!')

if __name__ == '__main__':
    sub_first()  #TPR@FPR=10e-4 0.9971
    sub_second() #TPR@FPR=10e-4 0.9991

from argparse import ArgumentParser
from configparser import ConfigParser
from importlib import import_module
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 核心库
import numpy as np
import torch as t

# 自定义库
from data.utils import getwhichpart


if __name__ == '__main__':
    ## 1.读取配置
    # 1.1命令行参数
    '''
        --subject: 要测试的受试者范围，一个闭区间
        --config: deap
        --model: model里model的名字
        --extracted: 数据处理模式，用-连接不同的设置，顺序不敏感
            输入default为默认模式：微伏-pps都不减
            要改变某些设置，格式如v-pps_baseline
            min_max
            z_score
        --label: 要使用的标签
    '''
    parser = ArgumentParser(description='STHGNN')
    parser.add_argument('--subject', nargs=2, type=int, required=True)
    parser.add_argument('--config', nargs=1, type=str, required=True)
    parser.add_argument('--model', nargs=1, type=str, required=True)
    parser.add_argument('--extracted', nargs=1, type=str, required=True)
    parser.add_argument('--label', nargs=1, type=int, required=True)
    # 解析
    args = parser.parse_args()
    train_subject = args.subject
    config_path = "./config/" + args.config[0] + ".config"
    extracted_way = args.extracted[0]
    if args.config[0][:4] == 'deap':
        data_lib = import_module('data.process_deap')
        Dataset = data_lib.DeapDateset
        labels = 4
    else:
        data_lib = import_module('data.process_hci')
        Dataset = data_lib.HciDateset
        labels = 5
    label_num = args.label[0]
    # 1.2读取config文件
    config = ConfigParser()
    config.read(config_path)

    '''
        data_config: 用于配置数据处理部分的路径，直接传入process函数
        feature_config: 用于配置如何提取数据特征，直接传入process函数
        model_config: 用于配置模型参数，直接传入Net对象
        train_config: 用于配置训练参数，在该文件下载入
        dataset_config：包含处理dataset的进程数和相关系数阈值
    '''
    data_config = config['path']
    feature_config = config['feature']
    model_config = config['model']
    train_config = config['train']
    dataset_config = config['dataset']

    fold = int(train_config['fold'])

    ## 2.准备数据
    try:
        assert extracted_way == 'none' or set(extracted_way.split('-')) <= {'', 'default', 'v', 'mv', 'z_score',
                                                                            'min_max',
                                                                            'pps_baseline'}
    except AssertionError:
        print("extracted_way wrong")
        raise RuntimeError
    if extracted_way != 'none':
        data_lib.process_all(train_subject, extracted_way, data_config, feature_config)

    ## 3.k折交叉验证
    all_subject_history = {}
    all_subject_history['train confusion'] = t.zeros(args.subject[1] - args.subject[0] + 1, 3, 3)
    all_subject_history['train acc'] = t.zeros(args.subject[1] - args.subject[0] + 1, fold)
    all_subject_history['train epoch'] = t.zeros(args.subject[1] - args.subject[0] + 1, fold)
    all_subject_history['test confusion'] = t.zeros(args.subject[1] - args.subject[0] + 1, 3, 3)
    all_subject_history['test acc'] = t.zeros(args.subject[1] - args.subject[0] + 1, fold)
    all_subject_history['test epoch'] = t.zeros(args.subject[1] - args.subject[0] + 1, fold)

    num_node = int(model_config['num_node'])
    sample_feature_num = int(model_config['sample_feature_num'])
    for i in range(args.subject[0], args.subject[1] + 1):
        if args.config[0][:3] == 'hci' and i in [9, 12, 15]:
            continue
        dataset = Dataset(i, data_config, dataset_config).shuffle()
        acc_list = []
        for j in range(fold):
            train_data, test_data = getwhichpart(dataset, j,label_num)
            x_train = np.zeros((len(train_data),4*num_node))
            x_test = np.zeros((len(test_data),4*num_node))
            y_train = np.zeros((len(train_data)))
            y_test = np.zeros((len(test_data)))
            for k in range(len(train_data)):
                x_train[k] = np.reshape(train_data[k].FS.numpy(), (4*num_node))
                y_train[k] = train_data[k].Y[label_num].numpy()
            for k in range(len(test_data)):
                x_test[k] = np.reshape(test_data[k].FS.numpy(), (4*num_node))
                y_test[k] = test_data[k].Y[label_num].numpy()
            #调用svm
            model = SVC(kernel='rbf',C=10)
            model.fit(x_train,y_train)
            predicts = model.predict(x_test)
            acc_list.append(accuracy_score(y_test,predicts))
        print("subject: ",i," acc_list:",acc_list)
        acc_list = np.array(acc_list)
        print("average of ",i," ",acc_list.mean())





"""Eval"""
import os
import time
import datetime
import glob
import numpy as np
import mindspore.nn as nn

from mindspore import Tensor, context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

from src.utils.logging import get_logger
from src.vgg import vgg16_sentiment, Vgg_sentiment
from src.dataset import vgg_create_dataset
from src.dataset import classification_dataset
from src.dataset import create_dataset

from model_utils.moxing_adapter import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_rank_id, get_device_num

from model_utils.config import get_config
from fine_tune import DenseHead, cfg
import pickle


class ParameterReduce(nn.Cell):
    """ParameterReduce"""

    def __init__(self):
        super(ParameterReduce, self).__init__()
        self.cast = P.Cast()
        self.reduce = P.AllReduce()

    def construct(self, x):
        one = self.cast(F.scalar_to_tensor(1.0), mstype.float32)
        out = x * one
        ret = self.reduce(out)
        return ret


def get_top5_acc(top5_arg, gt_class):
    sub_count = 0
    for top5, gt in zip(top5_arg, gt_class):
        if gt in top5:
            sub_count += 1
    return sub_count


def modelarts_pre_process():
    '''modelarts pre process function.'''

    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if config.device_target == "GPU":
            init()
            device_id = get_rank()
            device_num = get_group_size()
        elif config.device_target == "Ascend":
            device_id = get_device_id()
            device_num = get_device_num()
        else:
            raise ValueError("Not support device_target.")

        # Each server contains 8 devices as most.
        if device_id % min(device_num, 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(device_id, zip_file_1, save_dir_1))

    config.log_path = os.path.join(config.output_path, config.log_path)


def run_eval(attack_method):
    dataset_name = 'warship'
    """run eval"""
    config.per_batch_size = config.batch_size
    config.image_size = list(map(int, config.image_size.split(',')))
    config.rank = get_rank_id()
    config.group_size = get_device_num()

    _enable_graph_kernel = config.device_target == "GPU"
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=_enable_graph_kernel,
                        device_target=config.device_target, save_graphs=False)
    if os.getenv('DEVICE_ID', "not_set").isdigit() and config.device_target == "Ascend":
        context.set_context(device_id=int(os.getenv('DEVICE_ID')))

    config.outputs_dir = os.path.join(config.log_path,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    config.logger = get_logger(config.outputs_dir, config.rank)
    config.logger.save_args(config)

    if config.dataset == "cifar10":
        net = vgg16_sentiment(num_classes=config.num_classes, args=config)
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        model = Model(net, loss_fn=loss, metrics={'acc'})
        param_dict = load_checkpoint(config.pre_trained)
        load_param_into_net(net, param_dict)
        net.set_train(False)
        dataset = vgg_create_dataset(config.data_dir, config.image_size, config.per_batch_size, training=False)
        res = model.eval(dataset)
        print("result: ", res)
    else:
        # network
        config.logger.important_info('start create network')
        if os.path.isdir(config.pre_trained):
            models = list(glob.glob(os.path.join(config.pre_trained, '*.ckpt')))
            if config.graph_ckpt:
                f = lambda x: -1 * int(os.path.splitext(os.path.split(x)[-1])[0].split('-')[-1].split('_')[0])
            else:
                f = lambda x: -1 * int(os.path.splitext(os.path.split(x)[-1])[0].split('_')[-1])
            config.models = sorted(models, key=f)
        else:
            config.models = [config.pre_trained, ]
        for model in ['/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image_mindspore/save_models/0-1_109_warship.ckpt']:
            print(model)
            features_clean = []
            features_adv = []
            dataset_clean = classification_dataset(
                '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/VGG16/{}/ori'.format(
                    attack_method),
                config.image_size, config.per_batch_size, mode='eval')
            dataset_adv = classification_dataset(
                '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/VGG16/{}/adv'.format(
                    attack_method),
                config.image_size, config.per_batch_size, mode='eval')
            network = vgg16_sentiment(config.num_classes, config, phase="test")
            eval_dataloader_clean = dataset_clean.create_tuple_iterator(output_numpy=True)
            eval_dataloader_adv = dataset_adv.create_tuple_iterator(output_numpy=True)

            # pre_trained
            load_param_into_net(network, load_checkpoint(model))
            network.add_flags_recursive(fp16=True)

            network.set_train(False)
            t_end = time.time()
       
            for data, gt_classes in eval_dataloader_clean:
                for d in data:
                    d = np.expand_dims(d, axis=0)
                    feature = []
                    x = Tensor(d, mstype.float32)
                    for i in range(len(network.layers)):
                        x = network.layers[i](x)
                        if i == 5 or i == 14 or i == 35:
                            feature.append(x.asnumpy())

                    features_clean.append(feature)
            print(len(features_clean))
            with open('features/clean/clean_features_{}_{}.pkl'.format(dataset_name,attack_method), 'wb') as file:
                pickle.dump(features_clean, file)
            count = 0
            for data, gt_classes in eval_dataloader_adv:
                # output = network(Tensor(data, mstype.float32))
                # print(network.layers)
                if count < len(features_clean):
                    for d in data:
                        d = np.expand_dims(d, axis=0)
                        feature = []
                        x = Tensor(d, mstype.float32)
                        for i in range(len(network.layers)):
                            x = network.layers[i](x)
                            if i == 5 or i == 14 or i == 35:
                                feature.append(x.asnumpy())
                        features_adv.append(feature)
                        count += 1
            print(len(features_adv))
            with open('features/adv/adv_features_{}_{}.pkl'.format(dataset_name,attack_method), 'wb') as file:
                pickle.dump(features_adv, file)


if __name__ == "__main__":
    attack_method = 'PatchAttack'
    run_eval(attack_method)

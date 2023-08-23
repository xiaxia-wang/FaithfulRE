import argparse
import configparser
import os
import time

import torch.nn

from data import Data
from experiment_gpu import Experiment, read_training_time_from_file
from model import TransformerLearner


class Option(object):
    def __init__(self, arguments):
        self.__dict__ = arguments

    def save(self):
        self.dir = os.path.join(self.exps_dir, self.exp_name)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        with open(os.path.join(self.dir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))


def main(data_dir, exps_dir, model, process):
    parser = argparse.ArgumentParser(description="Experiment setup")
    parser.add_argument('--model', default=model, type=str)
    # misc
    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument('--no_train', default=False, action="store_true")
    parser.add_argument('--rule_thr', default=1e-2, type=float)
    parser.add_argument('--no_preds', default=False, action="store_true")
    parser.add_argument('--get_vocab_embed', default=False, action="store_true")
    parser.add_argument('--exps_dir', default=exps_dir, type=str)
    parser.add_argument('--exp_name', default=data_dir.split("/")[-1], type=str)
    # data property
    parser.add_argument('--data_dir', default=data_dir, type=str)
    # parser.add_argument('--resplit', default=False, action="store_true")
    parser.add_argument('--no_link_percent', default=0., type=float)
    parser.add_argument('--no_extra_facts', default=False, action="store_true")
    # model architecture
    parser.add_argument('--num_step', default=4, type=int)
    parser.add_argument('--num_layer', default=1, type=int)
    parser.add_argument('--rank', default=3, type=int)
    parser.add_argument('--rnn_state_size', default=128, type=int)
    parser.add_argument('--query_embed_size', default=128, type=int)
    # optimization
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--min_epoch', default=5, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--no_norm', default=False, action="store_true")
    parser.add_argument('--thr', default=1e-20, type=float)
    parser.add_argument('--dropout', default=0., type=float)
    # evaluation
    parser.add_argument('--min_beta', default=1e-10, type=int)

    option = Option(vars(parser.parse_args()))

    data = Data(option.data_dir, option.seed, option.no_extra_facts, True)
    print("Data prepared.")
    # print(data.num_relation)
    # print(data.query_for_rules)

    option.num_entity = data.num_entity
    option.num_operator = data.num_operator
    option.num_query = data.num_query

    option.dir = os.path.join(option.exps_dir, option.exp_name)
    if process == "train":
        option.save()
        print("Option saved at: ", option.dir)
    elif process == "evaluate" or process == "analyze":
        option.no_norm = True # no normalize for prediction

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    model = TransformerLearner(option)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])  # no need for small batch_size
    # print("Model initialized.")
    '''
    TransformerLearner(
      (query_embeddings): Embedding(19, 128)
      (rank_list): ModuleList(
        (0-2): 3 x TransformerEncoder(
          (layers): ModuleList(
            (0-5): 6 x TransformerEncoderLayer(
              (self_attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
              )
              (linear1): Linear(in_features=128, out_features=2048, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
              (linear2): Linear(in_features=2048, out_features=128, bias=True)
              (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              (dropout1): Dropout(p=0.1, inplace=False)
              (dropout2): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
      (linear): Linear(in_features=128, out_features=19, bias=True)
      (softmax): Softmax(dim=-1)
      (predictor): Predictor()
    )
    '''

    data.batch_init(option.batch_size)
    experiment = Experiment(option, model, data)
    print("Experiment initialized.", flush=True)
    if process == "train":
        experiment.train()
    elif process == "evaluate":
        assert option.no_norm
        experiment.get_prediction_valid()
        experiment.get_prediction_test()
    elif process == "analyze":
        assert option.no_norm
        experiment.search_for_beta(option.min_beta, 3, 1e-8)
        experiment.get_test_scores()


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("../config.ini")

    datasets_dir = config["DEFAULT"]["datasets_dir"]
    eval_dir = config["DEFAULT"]["drum_gpu_3"]
    for dir1 in ["fb237", "nell", "wn18rr"]:
        for dir2 in ["v1", "v2", "v3", "v4"]:
            srcDir = datasets_dir + "/" + dir1 + "_" + dir2
            start = time.time()
            main(srcDir, eval_dir, "drum", "train")
            end = time.time()
            print("Training time: ", round(end - start, 3), flush=True)

    # read_training_time_from_file("./out-drum-step3.txt")

    # eval_dir = config["DEFAULT"]["test_gpu_result"]
    # src_dir = datasets_dir + "/wn18rr_v4"
    # print(src_dir, " --> ", eval_dir)
    # main(src_dir, eval_dir, "drum", "train")

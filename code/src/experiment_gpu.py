import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import TransformerLearner

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Experiment():
    def __init__(self, option, model: TransformerLearner, data):
        self.option = option
        self.model = model.to(device)
        self.data = data
        self.early_stop = False
        self.epoch = 0
        self.criterion = nn.CrossEntropyLoss().to(device)  # same as NLLLoss
        self.optimizer = optim.Adam(self.model.parameters(), lr=option.learning_rate)

        self.model_base_dir = os.path.join(option.dir, "models")

    def train(self):
        epoch_loss, valid_loss, test_loss = [], [], []
        while self.epoch < self.option.max_epoch and not self.early_stop:
            # self.data.shuffle()
            self.model.train()
            print("========", "Epoch ", self.epoch, "========", flush=True)
            epoch_loss.append(self.train_epoch())

            self.model.eval()
            with torch.no_grad():
                valid_loss.append(self.valid_epoch())
                if self.epoch >= self.option.min_epoch and valid_loss[-1] > valid_loss[-2]:
                    self.early_stop = True
                    print("Early stop at epoch", self.epoch, flush=True)
                test_loss.append(self.test_epoch())

            self.save(self.epoch)
            self.epoch += 1


    def train_epoch(self):
        epoch_loss = 0.0
        for batch in range(self.data.num_batch_train):
            (qq, hh, tt), mdb = self.data.next_train()
            # print(len(qq), len(hh), len(tt)) # 128 = 64 * 2， included reverse
            prediction, _ = self.model(qq, tt, mdb) # time-consuming
            # print(prediction.shape)
            # torch.Size([128, 10945])
            # targets = F.one_hot(torch.tensor(hh), num_classes=self.option.num_entity)
            targets = torch.tensor(hh, device=device)

            self.optimizer.zero_grad() # clear the existing gradients
            loss = self.criterion(prediction, targets)
            # print("loss: ", loss.item())
            epoch_loss += loss.item()
            # time0 = time.time()
            loss.backward()
            # for name, weight in self.model.named_parameters():
            #     if weight.requires_grad:
            #         print(name, weight.grad.mean(), weight.grad.min(), weight.grad.max())
            self.optimizer.step()
            # time1 = time.time()
            # print(round(time1 - time0, 2))
        print("Train loss: ", epoch_loss, flush=True)
        return epoch_loss

    def valid_epoch(self):
        epoch_loss = 0.0
        for batch in range(self.data.num_batch_valid):
            (qq, hh, tt), mdb = self.data.next_valid()
            prediction, _ = self.model(qq, tt, mdb)

            # targets = F.one_hot(torch.tensor(hh), num_classes=self.option.num_entity)
            targets = torch.tensor(hh, device=device)

            loss = self.criterion(prediction, targets)
            epoch_loss += loss.item()

        print("Valid loss: ", epoch_loss, flush=True)
        return epoch_loss

    def test_epoch(self):
        epoch_loss = 0.0
        for batch in range(self.data.num_batch_test):
            (qq, hh, tt), mdb = self.data.next_test()
            prediction, _ = self.model(qq, tt, mdb)

            # targets = F.one_hot(torch.tensor(hh), num_classes=self.option.num_entity)
            targets = torch.tensor(hh, device=device)

            loss = self.criterion(prediction, targets)
            epoch_loss += loss.item()

        print("Test loss: ", epoch_loss, flush=True)
        return epoch_loss

    def save(self, epoch):
        if not os.path.exists(self.model_base_dir):
            os.makedirs(self.model_base_dir)
        file_path = self.model_base_dir + "/" + str(epoch) + ".pt"
        torch.save(self.model.state_dict(), file_path)
        print("Model", epoch, "saved at", file_path, flush=True)
        # model.load_state_dict(torch.load(model_save_dir))

    def test_loading_model(self, epoch=-1):
        if epoch == -1:
            epoch = len(os.listdir(self.model_base_dir)) - 1
        file_path = self.model_base_dir + "/" + str(epoch) + ".pt"
        self.model.load_state_dict(torch.load(file_path))
        print("Model is loaded from", file_path, flush=True)
        self.model.eval()
        with torch.no_grad():
            self.valid_epoch()
            self.test_epoch()

    def get_attention(self, epoch=-1):
        if epoch == -1:
            epoch = len(os.listdir(self.model_base_dir)) - 1
        file_path = self.model_base_dir + "/" + str(epoch) + ".pt"
        self.model.load_state_dict(torch.load(file_path))
        print("Model is loaded from", file_path, flush=True)
        self.model.eval()
        with torch.no_grad():
            (qq, hh, tt), mdb = self.data.get_attention_batch()
            _, attention_list = self.model(qq, tt, mdb)
            # print(len(attention_list), len(attention_list[0]), attention_list[0][0].shape)
            # 3 3 torch.Size([19, 19])  self.rank, self.num_step, (batch_size, num_operator)
            # print(torch.sum(attention_list[0][0], dim=1))
            return attention_list

    def get_prediction(self, q, h, t, epoch=-1):
        if epoch == -1:
            epoch = len(os.listdir(self.model_base_dir)) - 1
        file_path = self.model_base_dir + "/" + str(epoch) + ".pt"
        self.model.load_state_dict(torch.load(file_path))
        print("Model is loaded from", file_path, flush=True)
        self.model.eval()
        with torch.no_grad():
            (qq, hh, tt), mdb = ((q, q), (h, h), (t, t)), self.data.matrix_db_test
            prediction, attention = self.model(qq, tt, mdb)
            print(prediction[0][h].item())

    def get_prediction_valid(self, epoch=-1):
        if epoch == -1:
            epoch = len(os.listdir(self.model_base_dir)) - 1
        file_path = self.model_base_dir + "/" + str(epoch) + ".pt"
        self.model.load_state_dict(torch.load(file_path))
        print("Model is loaded from", file_path, flush=True)
        self.model.eval()
        with torch.no_grad():
            '''loop over the validation dataset that contains positive and negative examples'''
            f = open(os.path.join(self.option.dir, "valid_preds_and_probs_all.txt"), "w")
            for batch in range(self.data.num_batch_valid_evaluate):
                (qq, hh, tt), mdb = self.data.next_valid_evaluate()
                prediction, _ = self.model(qq, tt, mdb)
                # print(prediction.shape) # torch.Size([128, 10945])
                for i, (q, h, t) in enumerate(zip(qq, hh, tt)):
                    to_write = str(q) + " " + str(h) + " " + str(t) # beginning of the line
                    h_value = str(prediction[i][h].item())
                    preds = []
                    mask = prediction[i] > self.option.min_beta
                    inds = torch.nonzero(mask)
                    for k in inds:
                        preds.append([k.item(), prediction[i][k].item()])
                    preds.sort(key=lambda x: x[1], reverse=True)
                    preds_write = ",".join([str(p[0]) + " " + str(p[1]) for p in preds])
                    f.write(to_write + "\t" + h_value + "\t" + preds_write + "\n")
                    f.flush()
                print("batch: ", batch)
            f.close()

    def get_prediction_test(self, epoch=-1):
        if epoch == -1:
            epoch = len(os.listdir(self.model_base_dir)) - 1
        file_path = self.model_base_dir + "/" + str(epoch) + ".pt"
        self.model.load_state_dict(torch.load(file_path))
        print("Model is loaded from", file_path, flush=True)
        self.model.eval()
        with torch.no_grad():
            '''loop over the test dataset that contains positive and negative examples'''
            f = open(os.path.join(self.option.dir, "test_preds_and_probs_all.txt"), "w")
            for batch in range(self.data.num_batch_test_evaluate):
                (qq, hh, tt), mdb = self.data.next_test_evaluate()
                prediction, _ = self.model(qq, tt, mdb)
                # print(prediction.shape) # torch.Size([128, 10945])
                for i, (q, h, t) in enumerate(zip(qq, hh, tt)):
                    to_write = str(q) + " " + str(h) + " " + str(t) # beginning of the line
                    h_value = str(prediction[i][h].item())
                    preds = []
                    mask = prediction[i] > self.option.min_beta
                    inds = torch.nonzero(mask)
                    for k in inds:
                        preds.append([k.item(), prediction[i][k].item()])
                    preds.sort(key=lambda x: x[1], reverse=True)
                    preds_write = ",".join([str(p[0]) + " " + str(p[1]) for p in preds])
                    f.write(to_write + "\t" + h_value + "\t" + preds_write + "\n")
                    f.flush()
                print("batch: ", batch)
            f.close()

    def search_for_beta(self, low, high, gap):
        _, preds = self.read_preds_and_probs_file(
            os.path.join(self.option.dir, "valid_preds_and_probs_all.txt"))
        truths = self.data.valid_evaluate_label
        assert len(truths) * 2 == len(preds)
        while high - low >= gap:
            print("Search for beta in: ", low, "to", high)
            beta_vals = []
            f1_scores = []
            for i in range(0, 11):
                beta = round(low + (high - low) * i / 10, 9)
                beta_vals.append(beta)
                tp, fp, tn, fn = 0, 0, 0, 0
                for qht, v in truths.items():
                    if v == 1:
                        if preds[qht] > beta:
                            tp += 1
                        else:
                            fn += 1
                    elif v == 0:
                        if preds[qht] > beta:
                            fp += 1
                        else:
                            tn += 1
                prec = precision(tp, fp, tn, fn)
                rec = recall(tp, fp, tn, fn)
                accu = accuracy(tp, fp, tn, fn)
                f1 = f1score(tp, fp, tn, fn)
                print(beta, ": ", "precision: ", prec, "recall: ", rec, "accuracy:", accu, "f1:", f1, flush=True)
                f1_scores.append(f1)
            ind = f1_scores.index(max(f1_scores))
            print("beta: ", beta_vals[ind], "max_f1: ", max(f1_scores), flush=True)
            if ind == 0:
                high = low + (high - low) * 0.1
            elif ind == 10:
                low = low + (high - low) * 0.9
            else:
                low_new, high_new = low + (high - low) * (ind - 1.) / 10, low + (high - low) * (ind + 1.) / 10
                low, high = low_new, high_new
            print("=" * 32)
        last_ind = -1
        for i in range(len(f1_scores) - 1, -1, -1):
            if f1_scores[i] == max(f1_scores):
                last_ind = i
                break
        beta_ans = beta_vals[(ind + last_ind) // 2]
        print(ind, last_ind, beta_ans)
        with open(os.path.join(self.option.dir, "beta.txt"), "w") as f:
            f.write(str(beta_ans) + "\t" + str(max(f1_scores)) + "\n")
        return beta_ans

    def get_test_scores(self):
        beta = 0.0
        with open(os.path.join(self.option.dir, "beta.txt"), "r") as f:
            line = f.readline().rstrip().split("\t")
            beta = float(line[0])
        _, preds = self.read_preds_and_probs_file(
            os.path.join(self.option.dir, "test_preds_and_probs_all.txt"))
        truths = self.data.test_evaluate_label
        assert len(truths) * 2 == len(preds)
        tp, fp, tn, fn = 0, 0, 0, 0
        for qht, v in truths.items():
            if v == 1:
                if preds[qht] > beta:
                    tp += 1
                else:
                    fn += 1
            elif v == 0:
                if preds[qht] > beta:
                    fp += 1
                else:
                    tn += 1
        prec = precision(tp, fp, tn, fn)
        rec = recall(tp, fp, tn, fn)
        accu = accuracy(tp, fp, tn, fn)
        f1 = f1score(tp, fp, tn, fn)
        auc = self.get_test_auc(preds, truths)
        print(self.option.exp_name)
        print(beta, ": ",
              "precision: ", round(prec, 4), "recall: ", round(rec, 4), "accuracy:", round(accu, 4),
              "auc:", round(auc, 4), "f1:", round(f1, 4), flush=True)

    def get_test_auc(self, preds, truths):
        prec_vector = [0]
        recall_vector = [1]
        for beta in np.arange(0.01, 3, 0.01).tolist():
            tp, fp, tn, fn = 0, 0, 0, 0
            for qht, v in truths.items():
                if v == 1:
                    if preds[qht] > beta:
                        tp += 1
                    else:
                        fn += 1
                elif v == 0:
                    if preds[qht] > beta:
                        fp += 1
                    else:
                        tn += 1
            prec = precision(tp, fp, tn, fn)
            rec = recall(tp, fp, tn, fn)
            prec_vector.append(prec)
            recall_vector.append(rec)
        prec_vector.append(1)
        recall_vector.append(0)
        return auc(np.nan_to_num(prec_vector), np.nan_to_num(recall_vector))


    def read_preds_and_probs_file(self, file_name):
        results = []
        predictions = {}
        with open(file_name) as file:
            for line in file:
                line = line.rstrip().split("\t")
                # assert len(line) == 3
                triple = line[0].split(" ")
                q, h, t = int(triple[0]), int(triple[1]), int(triple[2])
                h_val = float(line[1])
                preds = []
                if len(line) == 3:
                    for pred_i in line[2].split(","):
                        k = pred_i.split(" ")
                        preds.append((int(k[0]), float(k[1])))
                results.append(((q, h, t), h_val, preds))
                # Each item: ((q, h, t), h_pred_val, [(p1, v1), (p2, v2), ..., (pk, vk)])
                predictions[(q, h, t)] = max(h_val, predictions.get((q, h, t), 0.0))
        return results, predictions


def read_training_time_from_file(file_name):
    with open(file_name, "r") as file:
        for line in file:
            line = line.rstrip()
            # print(line)
            if line.startswith("Option saved at"):
                dataset = line.split("/")[-1]
            if line.startswith("Training time:"):
                time = line.split()[2]
                # print(dataset, round(float(time)/60, 1))
                print(round(float(time)/60, 1))


def precision(tp, fp, tn, fn):
    value = 0
    try:
        value = tp / (tp + fp)
    except:
        value = float("NaN")
    finally:
        return value


def recall(tp, fp, tn, fn):
    value = 0
    try:
        value = tp / (tp + fn)
    except:
        value = float("NaN")
    finally:
        return value


def accuracy(tp, fp, tn, fn):
    value = 0
    try:
        value = (tp + tn) / (tp + fp + tn + fn)
    except:
        value = float("NaN")
    finally:
        return value


def f1score(tp, fp, tn, fn):
    value = 0
    try:
        value = tp / (tp + 0.5 * (fp + fn))
    except:
        value = float("NaN")
    finally:
        return value


def auc(precision_vector, recall_vector):
    return -1 * np.trapz(precision_vector, recall_vector)

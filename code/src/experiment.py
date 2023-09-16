import itertools
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import Learner


class Experiment():
    def __init__(self, option, model: Learner, data):
        self.option = option
        self.model = model
        self.data = data
        self.early_stop = False
        self.epoch = 0
        # self.criterion = nn.NLLLoss() # negative log likelihood loss
        self.criterion = nn.CrossEntropyLoss() # same as NLLLoss, but without LogSoftmax
        self.optimizer = optim.Adam(self.model.parameters(), lr=option.learning_rate)
        # similar performance for using Adam or AdamW

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
            targets = torch.tensor(hh)

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
            targets = torch.tensor(hh)

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
            targets = torch.tensor(hh)

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
        # feature_hooks = get_feature_by_hook(self.model) ########
        print("Model is loaded from", file_path, flush=True)
        self.model.eval()
        with torch.no_grad():
            self.valid_epoch()
            # print("attention shape: ", feature_hooks[0].feature.shape)
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

    def extract_rules_time_mm(self, thr=-1):
        # beta = 0.0
        # with open(os.path.join(self.option.dir, "beta.txt"), "r") as f:
        #     line = f.readline().rstrip().split("\t")
        #     beta = float(line[0])
        # if beta < self.option.min_beta:
        #     beta = self.option.min_beta
        # if thr == -1:
        #     thr = beta
        attention = self.get_attention()
        print("threshold: ", thr, flush=True)
        start = round(time.time() * 1000)
        kk = list(range(self.option.num_operator + 1))
        results_for_all_heads = []
        for h in range(self.option.num_operator // 2):  # h: head atom
            results = {}  # all rules with head atom being h
            for i in range(self.option.rank):
                curr_rules = {(): 1.0}
                for j in range(self.option.num_step - 1):
                    attn = attention[i][j].tolist()[h]
                    # print(len(attn))
                    updated_rules = {}
                    for k in kk: # enumerate over predicates, including identical (the last one)
                        for rule, val in curr_rules.items():
                            if k != self.option.num_operator:
                                rule = rule + (k,)
                            val *= attn[k]
                            if val > thr:
                                curr_val = updated_rules.get(rule, 0.0)
                                updated_rules[rule] = max(curr_val, val)
                    curr_rules = updated_rules
                # print(len(curr_rules)) # num_relations^3
                for rule, val in curr_rules.items():  # combine rules of different ranks
                    existing_val = results.get(rule, 0.0)
                    results[rule] = max(existing_val, val)
            results_for_all_heads.append(results)
        end = round(time.time() * 1000)
        time_cost = end - start
        print("time cost: ", time_cost, flush=True)
        return results_for_all_heads, time_cost

    def extract_rules_mm(self, thr=1e-3, topk=20):
        attention = self.get_attention()
        print("threshold: ", thr, flush=True)
        kk = list(range(self.option.num_operator + 1))
        results_for_all_heads = []
        for h in range(self.option.num_operator // 2):  # h: head atom
            results = {}  # all rules with head atom being h
            for i in range(self.option.rank):
                curr_rules = {(): 1.0}
                for j in range(self.option.num_step - 1):
                    attn = attention[i][j].tolist()[h]
                    # print(len(attn))
                    updated_rules = {}
                    for k in kk: # enumerate over predicates, including identical (the last one)
                        for rule, val in curr_rules.items():
                            if k != self.option.num_operator:
                                rule = rule + (k,)
                            val *= attn[k]
                            if val > thr:
                                curr_val = updated_rules.get(rule, 0.0)
                                updated_rules[rule] = max(curr_val, val)
                    curr_rules = updated_rules
                # print(len(curr_rules)) # num_relations^2
                for rule, val in curr_rules.items():  # combine rules of different ranks
                    existing_val = results.get(rule, 0)
                    results[rule] = max(existing_val, val)
            # sorted_rules = sorted(results.items(), key=lambda item: item[1], reverse=True)
            results_for_all_heads.extend((h, key, val) for key, val in results.items())
            results_for_all_heads = sorted(results_for_all_heads, key=lambda item: item[2], reverse=True)
        # print(results_for_all_heads[2])  # [((p1, p2, p3), value), ...] in descending order of value
        parser = self.data.parser["query"]
        if len(results_for_all_heads) < topk:
            topk = len(results_for_all_heads)
        for i in range(topk):
            rule = results_for_all_heads[i]
            head = parser[rule[0]]
            body = []
            for j in rule[1]:
                body.append(parser.get(j, ""))
            print(head, ": ", body[::-1], ": ", round(rule[2], 3), flush=True)
        return results_for_all_heads

    def extract_rules_time_sm(self, thr=1e-3):
        attention = self.get_attention()
        print("threshold: ", thr, flush=True)
        thr_aggregate = thr * self.option.rank
        # print(torch.sum(attention[0][0], dim=-1))
        start = round(time.time() * 1000)
        kk = list(range(self.option.num_operator + 1))
        results_for_all_heads = []
        for h in range(self.option.num_operator // 2):  # h: head atom
            results = []  # results[i]: rules with head atom h, from rank i
            iter_ranges = []
            for i in range(self.option.rank):
                curr_rules = {(): 1.0}
                for j in range(self.option.num_step - 1):
                    attn = attention[i][j].tolist()[h]
                    # print(len(attn))
                    updated_rules = {}
                    for k in kk: # enumerate over predicates, including identical (the last one)
                        for rule, val in curr_rules.items():
                            if k != self.option.num_operator:
                                rule = rule + (k,)
                            val *= attn[k]
                            if val > thr:
                                curr_val = updated_rules.get(rule, 0.0)
                                updated_rules[rule] = max(curr_val, val)
                    curr_rules = updated_rules
                # print(len(curr_rules)) # num_relations^3
                # results.append(sorted(curr_rules.items(), key=lambda item: item[1], reverse=True))
                results.append(tuple(curr_rules.items()))
                # element at i: rules from rank i
                iter_ranges.append(list(range(len(results[i]))))
            aggregated_rules = {}
            for index in itertools.product(*iter_ranges):
                value = 0.0
                rules = set()
                for i in range(self.option.rank):
                    subrule = results[i][index[i]]
                    value += subrule[1]
                    rules.add(subrule[0])
                if value > thr_aggregate:
                    rules = frozenset(rules)
                    existing_val = aggregated_rules.get(rules, 0.0)
                    aggregated_rules[rules] = max(existing_val, value)
            results_for_all_heads.append(aggregated_rules)
        end = round(time.time() * 1000)
        time_cost = end - start
        print("time cost: ", time_cost, flush=True)
        # print(len(results_for_all_heads))
        return results_for_all_heads, time_cost

    def extract_rules_sm(self, thr=1e-3, topk=10):
        attention = self.get_attention()
        print("threshold: ", thr, flush=True)
        thr_aggregate = thr * self.option.rank
        kk = list(range(self.option.num_operator + 1))
        results_for_all_heads = []
        for h in range(self.option.num_operator // 2):  # h: head atom
            results = []  # results[i]: rules with head atom h, from rank i
            iter_ranges = []
            for i in range(self.option.rank):
                curr_rules = {(): 1.0}
                for j in range(self.option.num_step - 1):
                    attn = attention[i][j].tolist()[h]
                    updated_rules = {}
                    for k in kk:  # enumerate over predicates, including identical (the last one)
                        for rule, val in curr_rules.items():
                            if k != self.option.num_operator:
                                rule = rule + (k,)
                            val *= attn[k]
                            if val > thr:
                                curr_val = updated_rules.get(rule, 0.0)
                                updated_rules[rule] = max(curr_val, val)
                    curr_rules = updated_rules
                results.append(tuple(curr_rules.items()))
                iter_ranges.append(list(range(len(results[i]))))
            aggregated_rules = {}
            for index in itertools.product(*iter_ranges):
                value = 0.0
                rules = set()
                for i in range(self.option.rank):
                    subrule = list(results[i][index[i]])
                    value += subrule[1]
                    rules.add(tuple(subrule[0][::-1]))
                if value > thr_aggregate:
                    rules = frozenset(rules)
                    existing_val = aggregated_rules.get(rules, 0.0)
                    aggregated_rules[rules] = max(existing_val, value)
            for final_rule, final_val in aggregated_rules.items():
                results_for_all_heads.append((h, tuple(final_rule), final_val))
                    # results_for_all_heads.append((h, tuple(rules), value))
        results_for_all_heads = sorted(results_for_all_heads, key=lambda item: item[2], reverse=True)
        # print(results_for_all_heads[2])  # (11, ((24, 16), (15, 5)), 0.3536216714642408)

        parser = self.data.parser["query"]
        if len(results_for_all_heads) < topk:
            topk = len(results_for_all_heads)
        for i in range(topk):
            rule = results_for_all_heads[i]
            head = parser[rule[0]]
            body = " | "
            for rank in rule[1]:
                for p in rank:
                    body += parser.get(p, "") + ", "
                body += " | "
            print(head, ": ", body, ": ", round(rule[2], 3), flush=True)
        return results_for_all_heads

    def extract_rule_time_drum(self, thr=1e-3):
        link_list = []
        for line in open(os.path.join(self.option.dir, "test_preds_and_probs_all.txt"), "r").readlines():
            case = line.split("\t")
            if float(case[1]) > thr:
                rht = case[0].split(" ")
                link_list.append((int(rht[0]), int(rht[1]), int(rht[2])))
        graph = defaultdict(set)  # tail: [(rel, head)]
        for r, h, t in self.data.test_facts:
            graph[t].add((r, h))
            graph[h].add((r + self.data.num_relation, t))
        # attention = self.get_attention()
        assert self.option.num_step == 4
        ##########################################################
        print("Model: ", self.option.dir)
        print("threshold: ", thr, flush=True)
        # print("number of all test_cases: ", len(link_list))
        start = round(time.time() * 1000)
        all_explanation = []
        # count = 0
        for r, h, t in link_list:  # loop over cases
            # count += 1
            # if count % 50 == 0:
            #     print(count)
            curr_exp = {}
            prev = set()
            prev.add((t,))
            for j in range(self.option.num_step - 1):
                curr = set()
                for p in prev:
                    for rr, hh in graph[p[-1]]:
                        curr.add(p + (rr, hh))
                    curr.add(p + (self.data.num_operator, p[-1]))
                prev = curr
            for p in prev:
                if p[-1] == h:
                    path = (p[1], p[3], p[5])
                    num = curr_exp.get(path, 0) + 1
                    curr_exp[path] = num
            all_explanation.append(tuple(curr_exp.items()))
        end = round(time.time() * 1000)
        time_cost = end - start
        print("time cost: ", time_cost, flush=True)
        return time_cost

    def compare_prediction_results(self, thr=1e-3):
        print("threshold: ", thr, flush=True)
        num_model = 0  # number of positive links predicted by the model
        test_links, predicted_links = {}, set()
        for line in open(os.path.join(self.option.dir, "test_preds_and_probs_all.txt"), "r").readlines():
            triple = line.strip().split("\t")[0].split()
            pso = (int(triple[0]), int(triple[1]), int(triple[2]))
            value = float(line.strip().split("\t")[1])
            test_links[pso] = value
            if pso in self.data.test and value > thr:
                num_model += 1
                predicted_links.add(pso)
        if num_model == 0:
            print("num_model = 0")
            return
        #################################################
        num_rules = 0
        prev_adj = defaultdict(set)  # tail: [(rel, head)]
        for r, h, t in self.data.test_facts:
            # if h == 1 or t == 1 or h == 0 or t == 0:
            #     print(h, r, t)
            adj_t, adj_h = prev_adj[t], prev_adj[h]
            adj_t.add((r, h))
            adj_h.add((r + self.data.num_relation, t))
            prev_adj[t] = adj_t
            prev_adj[h] = adj_h
        attention = self.get_attention()

        # print(list(prev_adj[1]))
        # print(list(prev_adj[2172]))
        # print(list(prev_adj[0]))
        for r, h, t in self.data.test:  # loop over cases
            for i in range(self.option.rank):  # loop over ranks
                pre_confidence = {t: 1.0}  # initiate
                for j in range(self.option.num_step - 1):
                    curr_confidence = {}
                    attn = attention[i][j].tolist()[r]
                    # print(attn)
                    for curr_t, curr_val in pre_confidence.items():
                        for rr, hh in prev_adj[curr_t]:
                            val_new = max(curr_confidence.get(hh, 0.0), curr_val * attn[rr])
                            if val_new > thr:
                                curr_confidence[hh] = val_new
                        val_new_t = max(curr_confidence.get(curr_t, 0.0), curr_val * attn[-1])
                        if val_new_t > thr:
                            curr_confidence[curr_t] = val_new_t
                    pre_confidence = curr_confidence
                    # print(pre_confidence)
                if pre_confidence.get(h, 0.0) > thr:
                    num_rules += 1
                    if (r, h, t) not in predicted_links:
                        print(r, h, t, ": ", test_links[(r, h, t)], pre_confidence.get(h), flush=True)
                        print("WRONG!")
                        return
                    break
        print(num_model, num_rules, round(num_rules/num_model, 4), flush=True)
        return num_model, num_rules

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
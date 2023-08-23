import os
from collections import Counter
import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Data(object):
    def __init__(self, folder, seed, no_extra_facts, use_gpu=False):
        np.random.seed(seed)
        self.seed = seed
        self.use_extra_facts = not no_extra_facts
        self.query_include_reverse = True

        self.relation_file = os.path.join(folder, "relations.txt")
        self.entity_file = os.path.join(folder, "entities.txt")

        self.relation_to_number, self.entity_to_number = self._numerical_encode()
        self.number_to_entity = {v: k for k, v in self.entity_to_number.items()}
        self.num_relation = len(self.relation_to_number)
        self.num_entity = len(self.entity_to_number)
        self.num_query = self.num_relation * 2

        self.test_file = os.path.join(folder, "test_positive.txt")
        self.train_file = os.path.join(folder, "train.txt")
        self.valid_file = os.path.join(folder, "valid.txt")

        self.train_facts_file = os.path.join(folder, "train_facts.txt")
        self.test_facts_file = os.path.join(folder, "test_facts.txt")

        self.train, self.num_train = self._parse_triplets(self.train_file)
        self.valid, self.num_valid = self._parse_triplets(self.valid_file)
        self.test, self.num_test = self._parse_triplets(self.test_file)

        self.train_facts, self.num_train_fact = self._parse_triplets(self.train_facts_file)
        self.test_facts, self.num_test_fact = self._parse_triplets(self.test_facts_file)
        if use_gpu:
            self.matrix_db_train = self._db_to_matrix_db_gpu(self.train_facts)
            self.matrix_db_valid = self._db_to_matrix_db_gpu(self.train_facts)
            self.matrix_db_test = self._db_to_matrix_db_gpu(self.test_facts)
        else:
            self.matrix_db_train = self._db_to_matrix_db(self.train_facts)
            self.matrix_db_valid = self._db_to_matrix_db(self.train_facts)
            self.matrix_db_test = self._db_to_matrix_db(self.test_facts)

        self.num_operator = 2 * self.num_relation

        # get rules for queries and their inverses appeared in train and test
        self.query_for_rules = list(set(list(zip(*self.train))[0]) |
                                    set(list(zip(*self.test))[0]) |
                                    set(list(zip(*self._augment_with_reverse(self.train)))[0]) |
                                    set(list(zip(*self._augment_with_reverse(self.test)))[0])
                                    )
        self.parser = self._create_parser()

        self.valid_evaluate, self.num_valid_evaluate, self.valid_evaluate_label = \
            self._parse_triplets_evaluate(os.path.join(folder, "valid_positive_negative_score.txt"))
        self.test_evaluate, self.num_test_evaluate, self.test_evaluate_label = \
            self._parse_triplets_evaluate(os.path.join(folder, "test_positive_negative_score.txt"))


    def _create_parser(self):
        """Create a parser that maps numbers to queries and operators given queries"""
        assert (self.num_query == 2 * len(self.relation_to_number) == 2 * self.num_relation)
        parser = {"query": {}, "operator": {}}
        number_to_relation = {value: key for key, value
                              in self.relation_to_number.items()}
        for key, value in self.relation_to_number.items():
            parser["query"][value] = key
            parser["query"][value + self.num_relation] = "inv_" + key
        for query in range(self.num_relation):
            d = {}
            for k, v in number_to_relation.items():
                d[k] = v
                d[k + self.num_relation] = "inv_" + v
            parser["operator"][query] = d
            parser["operator"][query + self.num_relation] = d
        return parser

    def _numerical_encode(self):
        relation_to_number = {}
        with open(self.relation_file) as f:
            for line in f:
                l = line.strip().split()
                assert (len(l) == 1)
                relation_to_number[l[0]] = len(relation_to_number)

        entity_to_number = {}
        with open(self.entity_file) as f:
            for line in f:
                l = line.strip().split()
                assert (len(l) == 1)
                entity_to_number[l[0]] = len(entity_to_number)
        return relation_to_number, entity_to_number

    def _parse_triplets(self, file):
        """Convert (head, relation, tail) to (relation, head, tail)"""
        output = []
        with open(file) as f:
            for line in f:
                l = line.strip().split("\t")
                assert (len(l) == 3)
                output.append((self.relation_to_number[l[1]],
                               self.entity_to_number[l[0]],
                               self.entity_to_number[l[2]]))
        return output, len(output)

    def _parse_triplets_evaluate(self, file):
        """Convert (head, relation, tail, score) in the file (xx_positive_negative_score) to (relation, head, tail)"""
        output = []
        labels = {}
        with open(file) as f:
            for line in f:
                l = line.strip().split("\t")
                assert (len(l) == 4)
                q = self.relation_to_number[l[1]]
                h = self.entity_to_number[l[0]]
                t = self.entity_to_number[l[2]]
                output.append((q, h, t))
                labels[(q, h, t)] = int(l[3])
        return output, len(output), labels

    def _db_to_matrix_db(self, db):
        ''' Modified according to the input of torch.sparse_coo_tensor(),
            Args: 1, a tensor of row- and col- indexes
                        e.g., [[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]]
                  2, a list of values, e.g., [1., 2., 3., 4., 5., 6.]
                  3, size=[3, 2]
        '''
        matrix_db = {r: ([[0], [0]], [0.], [self.num_entity, self.num_entity])
                     for r in range(self.num_relation)}
        for i, fact in enumerate(db):
            rel = fact[0]
            head = fact[1]
            tail = fact[2]
            value = 1.
            matrix_db[rel][0][0].append(head)
            matrix_db[rel][0][1].append(tail)
            matrix_db[rel][1].append(value)
        mdb = dict()
        for r, mtx in matrix_db.items():
            mdb[r] = torch.sparse_coo_tensor(torch.tensor(mtx[0]), mtx[1], mtx[2]).to_sparse_csr()
            mdb[r + self.num_relation] = torch.sparse_coo_tensor(torch.tensor((mtx[0][1], mtx[0][0])), mtx[1], mtx[2]).to_sparse_csr()
        return mdb

    def _db_to_matrix_db_gpu(self, db):
        ''' to_cuda:0
        '''
        matrix_db = {r: ([[0], [0]], [0.], [self.num_entity, self.num_entity])
                     for r in range(self.num_relation)}
        for i, fact in enumerate(db):
            rel = fact[0]
            head = fact[1]
            tail = fact[2]
            value = 1.
            matrix_db[rel][0][0].append(head)
            matrix_db[rel][0][1].append(tail)
            matrix_db[rel][1].append(value)
        mdb = dict()
        for r, mtx in matrix_db.items():
            mdb[r] = torch.sparse_coo_tensor(torch.tensor(mtx[0]), mtx[1], mtx[2], device=device)
            mdb[r + self.num_relation] = torch.sparse_coo_tensor(torch.tensor((mtx[0][1], mtx[0][0])), mtx[1], mtx[2], device=device)
        return mdb

    def _db_to_matrix_db_dense(self, db):
        ''' return dense matrices
        '''
        matrix_db = {r: ([[0], [0]], [0.], [self.num_entity, self.num_entity])
                     for r in range(self.num_relation)}
        for i, fact in enumerate(db):
            rel = fact[0]
            head = fact[1]
            tail = fact[2]
            value = 1.
            matrix_db[rel][0][0].append(head)
            matrix_db[rel][0][1].append(tail)
            matrix_db[rel][1].append(value)
        mdb = dict()
        for r, mtx in matrix_db.items():
            mdb[r] = torch.sparse_coo_tensor(torch.tensor(mtx[0]), mtx[1], mtx[2]).to_dense()
            mdb[r + self.num_relation] = torch.t(mdb[r])
        return mdb

    def batch_init(self, batch_size):
        self.batch_size = batch_size
        self.train_start = 0
        self.valid_start = 0
        self.test_start = 0
        self.valid_evaluate_start = 0
        self.test_evaluate_start = 0
        self.num_batch_train = int(self.num_train / batch_size) + 1
        self.num_batch_valid = int(self.num_valid / batch_size) + 1
        self.num_batch_test = int(self.num_test / batch_size) + 1
        self.num_batch_valid_evaluate = int(self.num_valid_evaluate / batch_size) + 1
        self.num_batch_test_evaluate = int(self.num_test_evaluate / batch_size) + 1

    def _augment_with_reverse(self, triplets):
        augmented = []
        for triplet in triplets:
            augmented += [triplet, (triplet[0] + self.num_relation,
                                    triplet[2],
                                    triplet[1])]
        return augmented

    def _next_batch(self, start, size, samples):
        assert (start < size)
        end = min(start + self.batch_size, size)
        next_start = end % size
        this_batch = samples[start:end]
        if self.query_include_reverse:
            this_batch = self._augment_with_reverse(this_batch)
        # this_batch_id = range(start, end)
        return next_start, this_batch, (start, end)

    def _triplet_to_feed(self, triplets):
        queries, heads, tails = zip(*triplets)
        return queries, heads, tails

    def next_test(self):
        self.test_start, this_batch, _ = self._next_batch(self.test_start,
                                                          self.num_test,
                                                          self.test)
        matrix_db = self.matrix_db_test
        return self._triplet_to_feed(this_batch), matrix_db

    def next_valid(self):
        self.valid_start, this_batch, _ = self._next_batch(self.valid_start,
                                                           self.num_valid,
                                                           self.valid)
        matrix_db = self.matrix_db_valid
        return self._triplet_to_feed(this_batch), matrix_db

    def next_train(self):
        self.train_start, this_batch, _ = self._next_batch(self.train_start,
                                                                       self.num_train,
                                                                       self.train)
        matrix_db = self.matrix_db_train
        return self._triplet_to_feed(this_batch), matrix_db

    def shuffle(self):
        np.random.shuffle(self.train)
        np.random.shuffle(self.valid)

    def get_attention_batch(self):
        qq = tuple(range(self.num_operator + 1))
        hh = tuple([0] * len(qq))
        tt = tuple([0] * len(qq))
        matrix_db = self.matrix_db_train
        return (qq, hh, tt), matrix_db

    def next_valid_evaluate(self):
        self.valid_evaluate_start, this_batch, _ = self._next_batch(self.valid_evaluate_start,
                                                                                self.num_valid_evaluate,
                                                                                self.valid_evaluate)
        matrix_db = self.matrix_db_train
        return self._triplet_to_feed(this_batch), matrix_db

    def next_test_evaluate(self):
        self.test_evaluate_start, this_batch, _ = self._next_batch(self.test_evaluate_start,
                                                                               self.num_test_evaluate,
                                                                               self.test_evaluate)
        matrix_db = self.matrix_db_test
        return self._triplet_to_feed(this_batch), matrix_db
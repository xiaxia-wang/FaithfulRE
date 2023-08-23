import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class Learner(nn.Module, object):  # pure on cpu
    def __init__(self, option):
        super(Learner, self).__init__()
        self.option = option
        torch.manual_seed(option.seed)

        self.query_embeddings = nn.Embedding(self.option.num_query + 1, self.option.query_embed_size)
        # print(self.query_embeddings.weight.shape, self.query_embeddings.weight.dtype, self.query_embeddings.weight.requires_grad)
        # torch.Size([19, 128]), torch.float32, True

        self.lstm_list = nn.ModuleList()
        for _ in range(self.option.rank):
            self.lstm_list.append(nn.LSTM(input_size=self.option.query_embed_size,  # 128
                                          hidden_size=self.option.rnn_state_size,  # 128
                                          num_layers=self.option.num_layer,  # 1
                                          bidirectional=True,
                                          batch_first=True))
        self.linear = nn.Linear(2 * self.option.rnn_state_size, self.option.num_operator + 1)
        self.softmax = nn.Softmax(dim=-1)
        self.predictor = Predictor(option)  # drum, by default
        if self.option.model == "smdrum":
            self.predictor = Predictor_SM(option)
        elif self.option.model == "mmdrum":
            self.predictor = Predictor_MM(option)

    def forward(self, qq, tt, mdb):
        queries = [[q] * (self.option.num_step - 1) + [self.option.num_query] for q in qq]
        queries_input = F.embedding(torch.tensor(queries), self.query_embeddings.weight)
        # print("queries_input.shape: ", queries_input.shape)
        # (batch_size * 2) * num_steps * (query_embed_size) e.g., (128, 3, 128)
        attention_list = []
        for i in range(self.option.rank):
            output, _ = self.lstm_list[i](queries_input)
            # output: torch.Size([128, 3, 256])
            split_output = [torch.squeeze(out) for out in torch.split(output, 1, dim=1)]
            attn_output = [self.softmax(self.linear(so)) for so in split_output]
            # print(len(attn_output), attn_output[0].shape)
            # 3 torch.Size([128, 19]) : self.num_step, (batch_size, num_operator)
            attention_list.append(attn_output)
        # print(len(attention_list), len(attention_list[0]), attention_list[0][0].shape)
        # 3 3 torch.Size([128, 19])  self.rank, self.num_step, (batch_size, num_operator)
        prediction = self.predictor(attention_list, tt, mdb)
        return prediction, attention_list


class Predictor(nn.Module, object):
    '''prediction layer'''

    def __init__(self, option):
        super(Predictor, self).__init__()
        self.option = option

    def forward(self, attention_list, tt, mdb):
        tails = F.one_hot(torch.tensor(tt), num_classes=self.option.num_entity)
        memory_list = []  # each element <--> i-th rank
        for i in range(self.option.rank):
            # memory_list.append(torch.unsqueeze(tails, dim=1))
            # torch.Size([128, 1, 10945])
            # memory_list.append(tails) ################
            memory_read = tails
            for t in range(self.option.num_step):
                attention = torch.split(attention_list[i][t], 1, dim=1)
                # torch.Size([128, 1])
                if t < self.option.num_step - 1:
                    db_results = []
                    memory_read_t = torch.t(memory_read).float()
                    for r in range(self.option.num_operator):
                        product = torch.mm(mdb[r], memory_read_t)
                        # print(product.shape) # torch.Size([10945, 128]) 128: batch_size
                        db_results.append(torch.t(product) * attention[r])

                    db_results.append(memory_read * attention[-1])
                    # each element: torch.Size([128, 10945]) 128: batch_size
                    if self.option.no_norm:
                        added_db_results = sum(db_results) # no normalize, used in analyze phase
                    else:
                        added_db_results = F.normalize(sum(db_results),
                                                       p=1,
                                                       eps=self.option.thr)
                    memory_read = added_db_results
            memory_list.append(memory_read)
        prediction = torch.sum(torch.stack(memory_list, dim=0), dim=0)
        return prediction


class Predictor_SM(nn.Module, object):
    '''prediction layer, SUM + MAX'''

    def __init__(self, option):
        super(Predictor_SM, self).__init__()
        self.option = option

    def forward(self, attention_list, tt, mdb):
        tails = F.one_hot(torch.tensor(tt), num_classes=self.option.num_entity)
        memory_list = []  # each element <--> i-th rank
        for i in range(self.option.rank):
            memory_read = tails
            for t in range(self.option.num_step):
                attention = torch.split(attention_list[i][t], 1, dim=1)
                if t < self.option.num_step - 1:
                    db_results = []
                    memory_read_t = torch.t(memory_read).float()
                    for r in range(self.option.num_operator):
                        # product = torch.mm(mdb[r], memory_read_t)
                        product = torch.sparse.mm(mdb[r], memory_read_t, "max")
                        db_results.append(torch.t(product) * attention[r])

                    db_results.append(memory_read * attention[-1])
                    # each element: torch.Size([128, 10945]) 128: batch_size
                    added_db_results = F.normalize(sum(db_results),
                                                   p=1,
                                                   eps=self.option.thr)
                    memory_read = added_db_results
            memory_list.append(memory_read)
        prediction = torch.sum(torch.stack(memory_list, dim=0), dim=0)
        return prediction


class Predictor_MM(nn.Module, object):
    '''prediction layer, SUM + SUM'''

    def __init__(self, option):
        super(Predictor_MM, self).__init__()
        self.option = option

    def forward(self, attention_list, tt, mdb):
        tails = F.one_hot(torch.tensor(tt), num_classes=self.option.num_entity)
        memory_list = []  # each element <--> i-th rank
        for i in range(self.option.rank):
            memory_read = tails
            for t in range(self.option.num_step):
                attention = torch.split(attention_list[i][t], 1, dim=1)
                if t < self.option.num_step - 1:
                    db_results = []
                    memory_read_t = torch.t(memory_read).float()
                    for r in range(self.option.num_operator):
                        # product = torch.mm(mdb[r], memory_read_t)
                        product = torch.sparse.mm(mdb[r], memory_read_t, "max")
                        db_results.append(torch.t(product) * attention[r])

                    db_results.append(memory_read * attention[-1])
                    added_db_results = F.normalize(sum(db_results),
                                                   p=1,
                                                   eps=self.option.thr)
                    memory_read = added_db_results
            memory_list.append(memory_read)
        # prediction = torch.sum(torch.stack(memory_list, dim=0), dim=0)
        prediction, _ = torch.max(torch.stack(memory_list, dim=0), dim=0)
        return prediction


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class TransformerLearner(nn.Module, object):  # use gpu
    def __init__(self, option):
        super(TransformerLearner, self).__init__()
        self.option = option
        torch.manual_seed(option.seed)
        torch.cuda.manual_seed(option.seed)

        self.query_embeddings = nn.Embedding(self.option.num_query + 1, self.option.query_embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.option.query_embed_size, nhead=8, batch_first=True)

        self.rank_list = nn.ModuleList()
        for _ in range(self.option.rank):
            self.rank_list.append(nn.TransformerEncoder(encoder_layer, num_layers=6))
        self.linear = nn.Linear(self.option.rnn_state_size, self.option.num_operator + 1)
        self.softmax = nn.Softmax(dim=-1)
        self.predictor = Predictor_GPU(option)

    def forward(self, qq, tt, mdb):
        queries = [[q] * (self.option.num_step - 1) + [self.option.num_query] for q in qq]
        queries_input = F.embedding(torch.tensor(queries, device=device), self.query_embeddings.weight)
        attention_list = []
        for i in range(self.option.rank):
            output = self.rank_list[i](queries_input)
            split_output = [torch.squeeze(out) for out in torch.split(output, 1, dim=1)]
            attn_output = [self.softmax(self.linear(so)) for so in split_output]
            attention_list.append(attn_output)
        # print(len(attention_list), len(attention_list[0]), attention_list[0][0].shape)
        # 3 3 torch.Size([128, 19])  self.rank, self.num_step, (batch_size, num_operator)
        prediction = self.predictor(attention_list, tt, mdb)
        return prediction, attention_list


class Predictor_GPU(nn.Module, object):

    def __init__(self, option):
        super(Predictor_GPU, self).__init__()
        self.option = option

    def forward(self, attention_list, tt, mdb):
        tails = F.one_hot(torch.tensor(tt, device=device), num_classes=self.option.num_entity)
        memory_list = []  # each element <--> i-th rank
        for i in range(self.option.rank):
            memory_read = tails
            for t in range(self.option.num_step):
                attention = torch.split(attention_list[i][t], 1, dim=1)
                if t < self.option.num_step - 1:
                    db_results = []
                    memory_read_t = torch.t(memory_read).float()
                    for r in range(self.option.num_operator):
                        product = torch.mm(mdb[r], memory_read_t)
                        db_results.append(torch.t(product) * attention[r])

                    db_results.append(memory_read * attention[-1])
                    if self.option.no_norm:
                        added_db_results = sum(db_results)  # no normalize, used in analyze phase
                    else:
                        added_db_results = F.normalize(sum(db_results),
                                                       p=1,
                                                       eps=self.option.thr)
                    memory_read = added_db_results
            memory_list.append(memory_read)
        prediction = torch.sum(torch.stack(memory_list, dim=0), dim=0)
        return prediction

class Hook:
    def __init__(self):
        self.feature = None

    def get_hook(self, module, feature_in, feature_out):
        '''
        Must have 3 input parameters as above: [module, feature_in, feature_out].
        module: some submodule in torch, e.g., Linear, Conv2d,
        feature_in: input of the module as a [tuple].
        feature_out: output of the module as a [tensor].
        Note: feature_in and feature_out have different types.
        '''
        self.feature = feature_out


def get_feature_by_hook(model):
    """
    Register the hook to nn.Softmax output, by iterating the model parameters with type
    The output values will be updated in every forward pass
    Note to call: feature_hooks = get_feature_by_hook(self.model)
    """
    feature_hooks = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Softmax):
            cur_hook = Hook()
            m.register_forward_hook(cur_hook.get_hook)
            feature_hooks.append(cur_hook)

    return feature_hooks


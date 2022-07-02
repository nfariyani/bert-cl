import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
nltk.download('punkt')
from transformers import BertModel
from pytorch_metric_learning import losses, miners


class BertCLModel(torch.nn.Module):
    def __init__(self, config, bert_version='bert-base-uncased'):
        super(BertCLModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_version, output_hidden_states=True)
        self.in_dim = self.bert.config.hidden_size # in_dim = 768
        self.pooling = config.pooling
        self.embedding = config.embedding
        self.hidden_dim = config.hidden_dim  # hidden_dim = 512
        self.out_dim = config.out_dim  # out_dim = 1
        if self.embedding == 'concat':
            self.trans1 = nn.Linear(self.in_dim * 2, self.hidden_dim, bias=True)
        else:
            self.trans1 = nn.Linear(self.in_dim, self.hidden_dim, bias=True)
        self.trans2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layer_out = nn.Linear(self.hidden_dim, 1, bias=True)
        self.bce_logits_loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        if config.loss == "NTXent":
            self.temperature = config.tau

    def forward(self, token_ids, input_mask):
        emb_in = self.get_bert_emb(token_ids, input_mask, self.pooling)
        batch_size = emb_in.shape[0]
        cl_loss = ContrastiveLoss(batch_size=batch_size, temperature=self.temperature, verbose=False)
        closs = cl_loss(emb_in)
        eloss = self.en_loss(emb_in)
        loss = closs + eloss

        return loss

    def get_bert_emb(self, token_ids, input_masks, pooling):
        # get the embedding from the last layer
        outputs = self.bert(input_ids=token_ids, attention_mask=input_masks)
        last_hidden_states = outputs.hidden_states[-1]

        if pooling is None:
            pooling = 'cls'
        if pooling == 'cls':
            pooled = last_hidden_states[:, 0, :]
        elif pooling == 'mean':
            pooled = last_hidden_states.sum(axis=1) / input_masks.sum(axis=-1).unsqueeze(-1)
        elif pooling == 'max':
            pooled = torch.max((last_hidden_states * input_masks.unsqueeze(-1)), axis=1)
            pooled = pooled.values
        del token_ids
        del input_masks
        return pooled

    def en_loss(self, trans_in):
        batch_size = trans_in.shape[0]
        n = int(batch_size / 4)
        m = int(batch_size / 2)
        z_in = F.normalize(trans_in, dim=1)
        labels = []

        for i in range(0, n):
            labels.extend([1.0 for j in range(m - (i + 1))])
            labels.extend([0.0 for k in range(m)])
            t_labels = torch.tensor(labels).to(trans_in.device)

        pred_all = torch.tensor([]).to(trans_in.device)

        for i in range(0, n):
            for j in range(i + 1, batch_size):
                # vec_cat = torch.cat([z_in[i], z_in[j]], dim=0)
                # print(vec_cat.shape)
                # embed_in_1 = torch.cat([embed_all, vec_cat], dim=0)
                emb_in_1 = z_in[i]
                emb_in_2 = z_in[j]
                if self.embedding == 'concat':
                    emb_in = torch.cat([emb_in_1, emb_in_2], dim=0) # tensor dimension size [2 * 768]
                elif self.embedding == 'subtract':
                    emb_in = torch.sub(emb_in_2 - emb_in_1)  # tensor dimension size [768]
                elif self.embedding == 'hadamard_cat':
                    emb_in = torch.mul(emb_in_1, emb_in_2)  # hadamard product vector1 and vector2
                    emb_in = torch.cat([emb_in, emb_in_1], dim=0)  # concatenates hadamard product with vector1
                    emb_in = torch.cat([emb_in, emb_in_2], dim=0)  # tensor dimension size [3 * 768]

                h1 = self.relu(self.trans1(emb_in))
                h2 = self.relu(self.trans2(h1))
                x = self.layer_out(h2)
                # concatenation
                pred_all = torch.cat([pred_all, x], dim=0)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(pred_all, t_labels)

        del labels, t_labels, pred_all
        return loss


class BertModel(torch.nn.Module):
    def __init__(self, config, bert_version='bert-base-uncased'):
        super(BertCLModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_version, output_hidden_states=True)
        self.in_dim = self.bert.config.hidden_size # in_dim = 768
        self.pooling = config.pooling
        self.embedding = config.embedding
        self.hidden_dim = config.hidden_dim  # hidden_dim = 512
        self.out_dim = config.out_dim  # out_dim = 1
        if self.embedding == 'concat':
            self.trans1 = nn.Linear(self.in_dim * 2, self.hidden_dim, bias=True)
        else:
            self.trans1 = nn.Linear(self.in_dim, self.hidden_dim, bias=True)
        self.trans2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layer_out = nn.Linear(self.hidden_dim, 1, bias=True)
        self.bce_logits_loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        if config.loss == "NTXent":
            self.temperature = config.tau

    def forward(self, token_ids, input_mask):
        emb_in = self.get_bert_emb(token_ids, input_mask, self.pooling)
        batch_size = emb_in.shape[0]
        cl_loss = ContrastiveLoss(batch_size=batch_size, temperature=self.temperature, verbose=False)
        closs = cl_loss(emb_in)
        eloss = self.en_loss(emb_in)
        loss = closs + eloss

        return loss

    def get_bert_emb(self, token_ids, input_masks, pooling):
        # get the embedding from the last layer
        outputs = self.bert(input_ids=token_ids, attention_mask=input_masks)
        last_hidden_states = outputs.hidden_states[-1]

        if pooling is None:
            pooling = 'cls'
        if pooling == 'cls':
            pooled = last_hidden_states[:, 0, :]
        elif pooling == 'mean':
            pooled = last_hidden_states.sum(axis=1) / input_masks.sum(axis=-1).unsqueeze(-1)
        elif pooling == 'max':
            pooled = torch.max((last_hidden_states * input_masks.unsqueeze(-1)), axis=1)
            pooled = pooled.values
        del token_ids
        del input_masks
        return pooled

    def en_loss(self, trans_in):
        batch_size = trans_in.shape[0]
        n = int(batch_size / 4)
        m = int(batch_size / 2)
        z_in = F.normalize(trans_in, dim=1)
        labels = []

        for i in range(0, n):
            labels.extend([1.0 for j in range(m - (i + 1))])
            labels.extend([0.0 for k in range(m)])
            t_labels = torch.tensor(labels).to(trans_in.device)

        pred_all = torch.tensor([]).to(trans_in.device)

        for i in range(0, n):
            for j in range(i + 1, batch_size):
                # vec_cat = torch.cat([z_in[i], z_in[j]], dim=0)
                # print(vec_cat.shape)
                # embed_in_1 = torch.cat([embed_all, vec_cat], dim=0)
                emb_in_1 = z_in[i]
                emb_in_2 = z_in[j]
                if self.embedding == 'concat':
                    emb_in = torch.cat([emb_in_1, emb_in_2], dim=0) # tensor dimension size [2 * 768]
                elif self.embedding == 'subtract':
                    emb_in = torch.sub(emb_in_2 - emb_in_1)  # tensor dimension size [768]
                elif self.embedding == 'hadamard_cat':
                    emb_in = torch.mul(emb_in_1, emb_in_2)  # hadamard product vector1 and vector2
                    emb_in = torch.cat([emb_in, emb_in_1], dim=0)  # concatenates hadamard product with vector1
                    emb_in = torch.cat([emb_in, emb_in_2], dim=0)  # tensor dimension size [3 * 768]

                h1 = self.relu(self.trans1(emb_in))
                h2 = self.relu(self.trans2(h1))
                x = self.layer_out(h2)
                # concatenation
                pred_all = torch.cat([pred_all, x], dim=0)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(pred_all, t_labels)

        del labels, t_labels, pred_all
        return loss

def tokenize_mask_clauses(clauses, max_seq_len, tokenizer):
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    mask_token = "[MASK]"
    # sentences = clauses.partition(str(". "))
    sentences = nltk.tokenize.sent_tokenize(clauses)
    tokens = [cls_token]
    for sent in sentences:
        tokens += tokenizer.tokenize(sent)
        tokens = tokens[:-1] + [sep_token]
    # print(tokens)
    # if tokens is longer than max_seq_len
    if len(tokens) >= max_seq_len:
        tokens = tokens[:max_seq_len - 2] + [sep_token]
    assert len(tokens) <= max_seq_len
    t_id = tokenizer.convert_tokens_to_ids(tokens)
    # add padding
    padding = [0] * (max_seq_len - len(t_id))
    i_mask = [1] * len(t_id) + padding
    t_id += padding
    return t_id, i_mask


def tokenize_mask(sentences, max_seq_len, tokenizer):
    token_ids = []
    input_mask = []
    for sent in sentences:
        t_id, i_mask = tokenize_mask_clauses(sent, max_seq_len, tokenizer)
        token_ids.append(t_id)
        input_mask.append(i_mask)
    return token_ids, input_mask


def convert_tuple_to_tensor(input_tuple, use_gpu=False):
    token_ids, input_masks = input_tuple
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    input_masks = torch.tensor(input_masks, dtype=torch.long)
    if use_gpu:
        token_ids = token_ids.to("cuda")
        input_masks = input_masks.to("cuda")
    return token_ids, input_masks


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature, verbose=True):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose

    def forward(self, emb_in):
        z_in = F.normalize(emb_in, dim=1)
        representations = z_in
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose:
            print("Similarity matrix\n", similarity_matrix, "\n")

        def log_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose:
                print(f"sim({i}, {j})={sim_i_j}")

            numerator = torch.exp(sim_i_j / self.temperature)
            if self.verbose:
                print(f"numerator", numerator)
                #  print(emb_in.device())
                #  print(torch.tensor([i]).device())
            one_for_not_i = torch.ones((self.batch_size, )).to(emb_in.device).scatter_(0, torch.tensor([i]).to(emb_in.device), 0.0)
            if self.verbose:
                print(f"1{{k!={i}}}", one_for_not_i)

            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )
            if self.verbose:
                print(f"Denominator", denominator)

            loss_ij = - torch.log(numerator / denominator)
            if self.verbose:
                print(f"loss({i}, {j})={loss_ij}\n")

            return loss_ij.squeeze(0)

        n = int(self.batch_size / 4)
        loss = 0.0
        for i in range(0, n):
            for j in range(i+1, n):
                loss += log_ij(i, j)

        return -2 / n * (n-1) * loss


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # reset counter

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



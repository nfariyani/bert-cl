import numpy as np
import torch.nn
import nltk
nltk.download('punkt')
from transformers import BertModel
from pytorch_metric_learning import losses, miners


class BertCLModel(torch.nn.Module):
    def __init__(self, config, bert_version='bert-base-uncased'):
        super(BertCLModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_version, output_hidden_states=True)
        self.miner = miners.MultiSimilarityMiner()
        if config.loss == "infonce":
            self.loss_fn = losses.NTXentLoss(temperature=config.tau)
        # elif config.loss == "triple":
            # self.loss_fn = losses.TripletMarginLoss(margin=config.margin)
        self.use_hard_pair = config.use_hard_pair

    def forward(self, token_ids, input_mask, labels=None):
        emb_in = self.get_bert_emb(token_ids, input_mask)
        batch_size = emb_in.shape[0]

        if labels is None:
            # index 0-3 for positive labels, and the rest for negatives
            pos_labels = torch.full([int(batch_size / 4)], 0, device=emb_in.device)
            neg_labels = torch.arange(1, int(batch_size * 3 / 4) + 1, device=emb_in.device)
            labels = torch.cat([pos_labels, neg_labels], dim=0)

        if self.use_hard_pair:
            hard_pairs = self.miner(emb_in, labels)
            loss = self.loss_fn(emb_in, labels, hard_pairs)
        else:
            loss = self.loss_fn(emb_in, labels)
        # print('loss: {:,}'.format(loss))
        return loss

    def get_bert_emb(self, token_ids, input_masks):
        # get the embedding from the last layer
        last_layer_emb = self.bert(input_ids=token_ids, attention_mask=input_masks).hidden_states[-1]
        emb_sent = last_layer_emb[:, :, -1]
        # print(emb_sent.shape)
        del token_ids
        del input_masks
        return emb_sent


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



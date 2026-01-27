import torch
import torch.nn.functional as F
from torch import nn
from mamba_ssm import Mamba
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import time


class Mamba4Rec_LSA_SD(SequentialRecommender):
    def __init__(self, config, dataset):
        super(Mamba4Rec_LSA_SD, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]

        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # --------- 蒸馏参数 ---------
        self.use_distill = getattr(config, "use_revit_distill", False)
        if self.use_distill:
            self.distill_layers = config.get("distill_layers", [0, 1, 2])
            self.distill_method = config.get("distill_method", "mse")
            self.distill_temperature = config.get("distill_temperature", 1.0)
            self.lambda_distill = config.get("lambda_distill", 1.0)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len, return_all_layers=False):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)

        all_layer_outputs = []
        for i in range(self.num_layers):
            item_emb = self.mamba_layers[i](item_emb)
            all_layer_outputs.append(item_emb)

        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)

        if return_all_layers:
            return seq_output, all_layer_outputs
        else:
            return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]

        if self.use_distill:
            seq_output, all_layers = self.forward(item_seq, item_seq_len, return_all_layers=True)
        else:
            seq_output = self.forward(item_seq, item_seq_len)
            all_layers = None

        # ---------- 主任务 loss ----------
        if self.loss_type == "BPR":
            pos_emb = self.item_embedding(pos_items)
            neg_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_emb, dim=-1)
            main_loss = self.loss_fct(pos_score, neg_score)
        else:
            logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
            main_loss = self.loss_fct(logits, pos_items)

        # ---------- 蒸馏 loss ----------
        distill_loss = 0.0
        if self.use_distill:
            teacher_feat = all_layers[-1].detach()
            for l_idx in self.distill_layers:
                student_feat = all_layers[l_idx]
                if self.distill_method == "cosine":
                    cos_sim = F.cosine_similarity(student_feat, teacher_feat, dim=-1)
                    distill_loss += (1 - cos_sim).mean()
                else:
                    distill_loss += F.mse_loss(student_feat, teacher_feat)

            if self.loss_type == "CE":
                T = self.distill_temperature
                s_log = F.log_softmax(logits / T, dim=-1)
                teacher_seq_output = self.gather_indexes(teacher_feat, item_seq_len - 1)
                teacher_logits = torch.matmul(teacher_seq_output, self.item_embedding.weight.transpose(0, 1))
                t_soft = F.softmax(teacher_logits / T, dim=-1)
                distill_loss += F.kl_div(s_log, t_soft, reduction="batchmean") * (T ** 2)

        total_loss = main_loss + self.lambda_distill * distill_loss
        return total_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

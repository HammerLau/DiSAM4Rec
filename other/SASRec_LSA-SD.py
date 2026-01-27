import torch
import torch.nn.functional as F
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class SASRec_LSA_SD(SequentialRecommender):
    r"""
    SASRec with self-distillation (LSA-SD).
    """

    def __init__(self, config, dataset):
        super(SASRec_LSA_SD, self).__init__(config, dataset)

        # ----------- 原有参数 -----------
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # ----------- 原有层定义 -----------
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # ----------- Loss -----------
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("loss_type must be BPR or CE")

        # ----------- 蒸馏参数 -----------
        self.use_distill = getattr(config, "use_revit_distill", False)
        if self.use_distill:
            self.distill_layers = config.get("distill_layers", [0, 1, 2])
            self.distill_method = config.get("distill_method", "mse")
            self.distill_temperature = config.get("distill_temperature", 1.0)
            self.lambda_distill = config.get("lambda_distill", 1.0)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len, return_all_layers=False):
        position_ids = torch.arange(item_seq.size(1), device=item_seq.device).unsqueeze(0).expand_as(item_seq)
        input_emb = self.item_embedding(item_seq) + self.position_embedding(position_ids)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        attention_mask = self.get_attention_mask(item_seq)
        all_layer_outputs = self.trm_encoder(input_emb, attention_mask, output_all_encoded_layers=True)
        seq_output = self.gather_indexes(all_layer_outputs[-1], item_seq_len - 1)

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

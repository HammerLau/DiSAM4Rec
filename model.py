from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from module import *


# # ================== 蒸馏 Loss ==================
# class DistillLossReViT(nn.Module):
#     def __init__(self, temperature=2.0, alpha=0.7):
#         super().__init__()
#         self.temperature = temperature
#         self.alpha = alpha
#         self.mse = nn.MSELoss()
#         self.kl = nn.KLDivLoss(reduction='batchmean')

#     def forward(self, student_feature, teacher_feature, student_logits=None, teacher_logits=None):
#         loss_feat = self.mse(student_feature, teacher_feature.detach())
#         if student_logits is not None and teacher_logits is not None:
#             T = self.temperature
#             s_log_softmax = F.log_softmax(student_logits / T, dim=-1)
#             t_softmax = F.softmax(teacher_logits / T, dim=-1)
#             loss_logits = self.kl(s_log_softmax, t_softmax.detach()) * (T ** 2)
#         else:
#             loss_logits = 0.0
#         return self.alpha * loss_feat + (1 - self.alpha) * loss_logits


# ================== MSADB 层 ==================
class MSADBLayers(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers, use_time=False, time_bins=None,
                 hidden_size=None, dataset=None):
        super().__init__()
        self.num_layers = num_layers
        self.use_time = use_time
        if use_time:
            self.time_bins = time_bins
            self.time_embedding = nn.Embedding(self.time_bins, hidden_size)
            self.time_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.gmamba = MSADB(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dataset=dataset)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, input_tensor, time_seq=None):
        seq_emb = input_tensor
        hidden_states = self.gmamba(seq_emb, timestamp=time_seq)  # 传入 timestamp
        if self.num_layers == 1:
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + seq_emb)
        hidden_states = self.ffn(hidden_states)
        return hidden_states


# ================== MSADB 模型 ==================
class DiSAM4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(DiSAM4Rec, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.config = config
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]
        self.dataset = dataset

        # --------- Embedding ---------
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        # --------- 可选时间 embedding ---------
        self.use_time_embedding = getattr(config, "use_time_embedding", False)
        if self.use_time_embedding:
            self.time_bins = getattr(config, "time_bins", 1000)
            self.time_embedding = nn.Embedding(self.time_bins, self.hidden_size)
            self.time_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # --------- Mamba 层 ---------
        self.mamba_layers = nn.ModuleList([
            MSADBLayers(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
                dataset=self.dataset
            ) for _ in range(self.num_layers)
        ])

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("loss_type must be BPR or CE")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # ================== MSADB forward ==================
    def forward(self, item_seq, item_seq_len, interaction=None, return_all_layers=False):
        item_emb = self.item_embedding(item_seq)
        seq_emb = self.dropout(self.LayerNorm(item_emb))

        time_field = getattr(self.config, "TIME_FIELD", None)
        if self.use_time_embedding and interaction is not None and time_field in interaction:
            time_seq = interaction[time_field].long()  # [B, L]
        else:
            time_seq = None

        all_layer_outputs = []
        for i in range(self.num_layers):
            seq_emb = self.mamba_layers[i](seq_emb, time_seq=time_seq)
            all_layer_outputs.append(seq_emb)

        seq_output = self.gather_indexes(seq_emb, item_seq_len - 1)

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

        seq_output, all_layers = self.forward(item_seq, item_seq_len, interaction=interaction, return_all_layers=True)

        # 主任务 loss
        if self.loss_type == "BPR":
            pos_emb = self.item_embedding(pos_items)
            neg_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_emb, dim=-1)
            main_loss = self.loss_fct(pos_score, neg_score)
        else:
            logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
            main_loss = self.loss_fct(logits, pos_items)

        # 蒸馏 loss
        distill_loss = 0.0
        if getattr(self.config, "use_revit_distill", False):
            teacher_feat = all_layers[-1].detach()
            for l_idx in self.config["distill_layers"]:
                student_feat = all_layers[l_idx]
                distill_method = getattr(self.config, 'distill_method', None)
                if distill_method == "cosine":
                    cos_sim = F.cosine_similarity(student_feat, teacher_feat, dim=-1)
                    distill_loss += (1 - cos_sim).mean()
                else:
                    distill_loss += F.mse_loss(student_feat, teacher_feat)

            if self.loss_type == "CE":
                T = self.config["distill_temperature"]
                logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
                s_log = F.log_softmax(logits / T, dim=-1)
                teacher_output = self.gather_indexes(teacher_feat, item_seq_len - 1)
                teacher_logits = torch.matmul(teacher_output, self.item_embedding.weight.transpose(0, 1))
                t_soft = F.softmax(teacher_logits / T, dim=-1)
                distill_loss += F.kl_div(s_log, t_soft, reduction="batchmean") * (T ** 2)

        lambda_distill = getattr(self.config, "lambda_distill", 1.0)
        total_loss = main_loss + lambda_distill * distill_loss
        return total_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len, interaction=interaction)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)

        if hasattr(self, "time_used_flag") and self.time_used_flag:
            print("[INFO] Time embedding was used in this prediction.")

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len, interaction=interaction)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores


# ================== MSADB ==================
class MSADB(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dataset=None):
        super(MSADB, self).__init__()
        from mamba_ssm import Mamba

        self.dense1 = nn.Linear(d_model, d_model)
        self.dense2 = nn.Linear(d_model, d_model)
        self.projection = nn.Linear(d_model, d_model)
        self.combining_weights = nn.Parameter(torch.tensor([0.1, 0.1, 0.8], dtype=torch.float32))
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        avg_user_inter = dataset.avg_actions_of_users if dataset else 10
        sparsity = dataset.sparsity if dataset else 0.999
        print(f"📊 avg_user_inter = {avg_user_inter:.2f}, sparsity = {sparsity:.4f}")
        if avg_user_inter < 20 or sparsity > 0.995:
            self.IREN = ShortConvGLU(d_model)
            self.FWEN = GumbelGate(d_model)
        else:
            self.IREN = Conv_GRU(d_model)
            self.FWEN = Dual_Activation_Gate(d_model)

    def forward(self, input_tensor, timestamp=None):
        short_interest = self.IREN(input_tensor)

        flipped_input = input_tensor.clone()
        flipped_input[:, :45, :] = input_tensor[:, :45, :].flip(dims=[1])

        mamba_output = self.mamba(input_tensor)
        mamba_output_f = self.mamba(flipped_input)

        gate_fwd, gate_bwd = self.FWEN(input_tensor)
        gated_mamba = gate_fwd * mamba_output + gate_bwd * mamba_output_f

        combined_states = (
                self.combining_weights[0] * short_interest +
                self.combining_weights[1] * mamba_output_f +
                self.combining_weights[2] * gated_mamba
        )
        return self.projection(combined_states)
        # return input_tensor


# ================== Sparse Branch ==================
class ShortConvGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv_proj = nn.Conv1d(d_model, d_model * 2, kernel_size=3, padding=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2, groups=d_model)
        self.residual_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        x_conv = self.conv_proj(x.transpose(1, 2))
        x_conv, gate = x_conv.chunk(2, dim=1)
        x_glu = x_conv * torch.sigmoid(gate)
        x_depth = self.depthwise_conv(x_glu)
        out = x_depth.transpose(1, 2)
        return self.residual_proj(out) + x 


class GumbelGate(nn.Module):
    def __init__(self, d_model, tau=0.5):
        super().__init__()
        self.tau = tau
        self.linear = nn.Linear(d_model, 2)

    def forward(self, x):
        logits = self.linear(x.mean(dim=1))
        gumbel_out = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)
        gate_fwd = gumbel_out[:, 0].unsqueeze(-1).unsqueeze(-1)
        gate_bwd = gumbel_out[:, 1].unsqueeze(-1).unsqueeze(-1)
        return gate_fwd, gate_bwd


# ================== Dense Branch ==================
class Conv_GRU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.gru = nn.GRU(d_model, d_model, num_layers=1, bias=False, batch_first=True)

    def forward(self, x):
        self.gru.flatten_parameters()
        x_conv = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        gru_out, _ = self.gru(x_conv)
        return gru_out


class Dual_Activation_Gate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear_sig = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(d_model, d_model)
        )
        self.linear_silu = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        gate = self.linear_sig(x) + self.linear_silu(x)
        gate = self.dropout(gate)
        return gate, 1 - gate


# ================== FeedForward ==================
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
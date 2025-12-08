import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, global_mean_pool


class FiLMLayer(nn.Module):
    def __init__(self, in_dim, w_dim):
        super().__init__()
        self.gamma = nn.Linear(w_dim, in_dim)
        self.beta = nn.Linear(w_dim, in_dim)

    def forward(self, x, w, batch=None):
        if w is None:
            return x
        if w.dim() == 1:
            w = w.view(1, -1)
        g = self.gamma(w)
        b = self.beta(w)
        if batch is None:
            g = g.repeat(x.size(0), 1)
            b = b.repeat(x.size(0), 1)
        else:
            g = g[batch]
            b = b[batch]
        return x * (1 + g) + b


class Policy(nn.Module):
    def __init__(self, config, device="cuda:0"):
        super(Policy, self).__init__()

        model_params = config.get("model", {})
        raw_feat_dim = model_params.get("raw_feat_dim", 11)
        embed_dim = model_params.get("embed_dim", 32)
        gcn_hidden = model_params.get("gcn_hidden", 64)
        num_heads = model_params.get("num_heads", 8)
        lstm_hidden = model_params.get("lstm_hidden", 64)

        # 设备（更健壮的处理，自动降级到CPU）
        try:
            self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 动作数/目标数（动作数改为从配置驱动）
        action_cfg = config.get("action_space", {})
        self.num_actions = int(action_cfg.get("num_rules", model_params.get("num_actions", 7)))
        if self.num_actions < 1:
            self.num_actions = 7
        # 目标维度由配置驱动（默认双目标 cmax+ttotal）
        active_objs = config.get("objectives_config", {}).get("active_objectives", ["cmax", "ttotal"]) 
        self.num_objectives = len(active_objs)
        self.lstm_hidden_dim = lstm_hidden

        assert gcn_hidden % num_heads == 0, "gcn_hidden must be divisible by num_heads"

        # --- GNN 编码 ---
        self.feat_embed = nn.Linear(raw_feat_dim, embed_dim)
        self.gcn1 = GCNConv(embed_dim, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.attn = TransformerConv(
            in_channels=gcn_hidden,
            out_channels=gcn_hidden // num_heads,
            heads=num_heads
        )
        self.film = FiLMLayer(in_dim=gcn_hidden, w_dim=self.num_objectives)

        # --- LSTM 记忆 ---
        self.lstm_cell = nn.LSTMCell(input_size=gcn_hidden, hidden_size=lstm_hidden)
        self.lstm_layer_norm = nn.LayerNorm(lstm_hidden)
        self.lstm_dropout = nn.Dropout(0.3)

        # --- ✅ 多目标 Dueling 架构 ---
        # 优势流 A(s,a): 输出 [A * num_objectives]
        self.advantage_stream = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, self.num_actions * self.num_objectives)
        )
        # 价值流 V(s): 输出 [num_objectives]
        self.value_stream = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, self.num_objectives)
        )

        self._init_weights()
        try:
            self.to(self.device)
        except Exception:
            self.device = torch.device('cpu')
            self.to(self.device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, node_features, edge_index, hidden_state, batch=None, w=None):
        """
        输入:
          - node_features: [N, raw_feat_dim]
          - edge_index: [2, E] (PyG COO)
          - hidden_state: (h, c) 其中 h/c: [B, lstm_hidden]
          - batch: [N] (PyG batch 索引); 若为 None，默认全零 => 单图
        输出:
          - q_vectors: [B, num_actions, num_objectives]  (多目标 Q 向量)
          - (h_new, c_new)
        """
        h_prev, c_prev = hidden_state

        

        # 1) GNN 图嵌入
        x = F.relu(self.feat_embed(node_features))
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.gcn2(x, edge_index))
        x = self.attn(x, edge_index)
        x = self.film(x, w, batch)

        # PyG 的 global_mean_pool 需要 batch 索引
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_embedding = global_mean_pool(x, batch)  # [B, gcn_hidden]

        # 2) LSTM 状态更新
        h_new, c_new = self.lstm_cell(graph_embedding, (h_prev, c_prev))
        h_new = self.lstm_layer_norm(h_new)
        if torch.isnan(c_new).any() or torch.isinf(c_new).any():
            print("[WARN] LSTM cell state contains NaN/Inf, resetting...")
            c_new = torch.zeros_like(c_new)
        else:
            c_new = torch.clamp(c_new, min=-1e6, max=1e6)
        h_new = self.lstm_dropout(h_new)

        # 3) 多目标 Dueling Q 向量
        state_value_vector = self.value_stream(h_new)  # [B, num_objectives]
        advantage_vectors_flat = self.advantage_stream(h_new)  # [B, A*num_objectives]
        advantage_vectors = advantage_vectors_flat.view(-1, self.num_actions, self.num_objectives)  # [B, A, num_objectives]

        q_vectors = state_value_vector.unsqueeze(1) + (advantage_vectors - advantage_vectors.mean(dim=1, keepdim=True))
        
        q_vectors = torch.nan_to_num(q_vectors, nan=0.0, posinf=1e6, neginf=-1e6)
        q_vectors = torch.clamp(q_vectors, min=-100.0, max=100.0)

        if os.environ.get("DEBUG_ASSERT", "1") == "1":
            assert q_vectors.dim() == 3, f"Q must be 3D [B, A, {self.num_objectives}]"
            assert q_vectors.size(1) == self.num_actions and q_vectors.size(2) == self.num_objectives, \
                f"Q shape mismatch, got {tuple(q_vectors.shape)}, expected (*,{self.num_actions},{self.num_objectives})"
            assert torch.isfinite(q_vectors).all(), "Q contains non-finite values"

        return q_vectors, (h_new, c_new)

    def init_hidden_state(self, batch_size=1):
        h0 = torch.zeros(batch_size, self.lstm_hidden_dim, device=self.device)
        c0 = torch.zeros(batch_size, self.lstm_hidden_dim, device=self.device)
        return (h0, c0)

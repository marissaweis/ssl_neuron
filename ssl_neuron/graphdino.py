# Attention and Block adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# DINO adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/dino.py

import copy
import torch
import torch.nn as nn
from typing import Any


class GraphAttention(nn.Module):
    """ Implements GraphAttention.

    Graph Attention interpolates global transformer attention
    (all nodes attend to all other nodes based on their
    dot product similarity) and message passing (nodes attend
    to their 1-order neighbour based on dot-product
    attention).

    Attributes:
        dim: Dimensionality of key, query and value vectors.
        num_heads: Number of parallel attention heads.
        bias: If set to `True`, use bias in input projection layers.
          Default is `False`.
        use_exp: If set to `True`, use the exponential of the predicted
          weights to trade-off global and local attention.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 bias: bool = False,
                 use_exp: bool = True) -> nn.Module:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        self.use_exp = use_exp

        self.qkv_projection = nn.Linear(dim, dim * num_heads * 3, bias=bias)
        self.proj = nn.Linear(dim * num_heads, dim)
        
        # Weigth to trade of local vs. global attention.
        self.predict_gamma = nn.Linear(dim, 2)
        # Initialize projection such that gamma is close to 1
        # in the beginning of training.
        self.predict_gamma.weight.data.uniform_(0.0, 0.01)

        
    @torch.jit.script
    def fused_mul_add(a, b, c, d):
        return (a * b) + (c * d)

    def forward(self, x, adj):
        B, N, C = x.shape # (batch x num_nodes x feat_dim)
        qkv = self.qkv_projection(x).view(B, N, 3, self.num_heads, self.dim).permute(0, 3, 1, 2, 4)
        query, key, value = qkv.unbind(dim=3) # (batch x num_heads x num_nodes x dim)

        attn = (query @ key.transpose(-2, -1)) * self.scale # (batch x num_heads x num_nodes x num_nodes)

        # Predict trade-off weight per node
        gamma = self.predict_gamma(x)[:, None].repeat(1, self.num_heads, 1, 1)
        if self.use_exp:
            # Parameterize gamma to always be positive
            gamma = torch.exp(gamma)

        adj = adj[:, None].repeat(1, self.num_heads, 1, 1)

        # Compute trade-off between local and global attention.
        attn = self.fused_mul_add(gamma[:, :, :, 0:1], attn, gamma[:, :, :, 1:2], adj)
        
        attn = attn.softmax(dim=-1)

        x = (attn @ value).transpose(1, 2).reshape(B, N, -1) # (batch_size x num_nodes x (num_heads * dim))
        return self.proj(x)
    
    
class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> nn.Module:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    """ Implements an attention block.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 bias: bool = False,
                 use_exp: bool = True,
                 norm_layer: Any = nn.LayerNorm) -> nn.Module:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GraphAttention(dim, num_heads=num_heads, bias=bias, use_exp=use_exp)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim=dim, hidden_dim=dim * mlp_ratio)

    def forward(self, x, a):
        x = self.norm1(x)
        x = self.attn(x, a) + x
        x = self.norm2(x)
        x = self.mlp(x) + x
        return x
    
    
class GraphTransformer(nn.Module):
    def __init__(self,
                 n_nodes: int = 200,
                 dim: int = 32,
                 depth: int = 5,
                 num_heads: int = 8,
                 mlp_ratio: int = 2,
                 feat_dim: int = 8,
                 num_classes: int = 1000,
                 pos_dim: int = 32,
                 proj_dim: int = 128,
                 use_exp: bool = True) -> nn.Module:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, dim))

        self.blocks = nn.Sequential(*[
            AttentionBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, use_exp=use_exp)
            for i in range(depth)])

        self.to_pos_embedding = nn.Linear(pos_dim, dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

        self.projector = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, num_classes)
        )

        self.to_node_embedding = nn.Sequential(
            nn.Linear(feat_dim, dim * 2),
            nn.ReLU(True),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, node_feat, adj, lapl):
        B, N, _ = node_feat.shape

        # Compute initial node embedding.
        x = self.to_node_embedding(node_feat)

        # Compute positional encoding
        pos_embedding_token = self.to_pos_embedding(lapl)

        # Add "classification" token
        cls_pos_enc = self.cls_pos_embedding.repeat(B, 1, 1)
        pos_embedding = torch.cat((cls_pos_enc, pos_embedding_token), dim=1)

        cls_tokens = self.cls_token.repeat(B, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add classification token entry to adjanceny matrix. 
        adj_cls = torch.zeros(B, N + 1, N + 1, device=node_feat.device)
        # TODO(test if useful)
        adj_cls[:, 0, 0] = 1.
        adj_cls[:, 1:, 1:] = adj

        x += pos_embedding
        for block in self.blocks:
            x = block(x, adj_cls)
        x = x[:, 0]
        x = self.mlp_head(x)

        return x, self.projector(x)


class ExponentialMovingAverage():
    """ Exponential moving average.

    Attributes:
        decay: Moving average decay parameter in [0., 1.] (float).
    """
    def __init__(self, decay: float):
        super().__init__()
        self.decay = decay
        assert (decay > 0.) and (decay < 1.), 'Decay must be in [0., 1.]'

    def update_average(
        self,
        previous_state: torch.Tensor,
        update: torch.Tensor,
        decay: float = None,
    ):
        if previous_state is None:
            return update
        if decay is not None:
            return previous_state * decay + (1 - decay) * update
        else:
            return previous_state * self.decay + (1 - self.decay) * update


def update_moving_average(ema_updater, teacher_model, student_model, decay=None):
    for student_params, teacher_params in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_weights, weight_update = teacher_params.data, student_params.data
        teacher_params.data = ema_updater.update_average(teacher_weights, weight_update, decay=decay)


class GraphDINO(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        num_classes: int = 1000,
        student_temp: float = 0.9,
        teacher_temp: float = 0.06,
        moving_average_decay: float = 0.999,
        center_moving_average_decay: float = 0.9,
    ):
        super().__init__()

        self.student_encoder = transformer
        self.teacher_encoder = copy.deepcopy(self.student_encoder)

        # Weights of teacher model are updated using an exponential moving
        # average of the student model. Thus, disable gradient update.
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

        self.teacher_ema_updater = ExponentialMovingAverage(moving_average_decay)

        self.register_buffer('teacher_centers', torch.zeros(1, num_classes))
        self.register_buffer('previous_centers',  torch.zeros(1, num_classes))

        self.teacher_centering_ema_updater = ExponentialMovingAverage(center_moving_average_decay)

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    def compute_loss(self, teacher_logits, student_logits, eps = 1e-20):
        teacher_logits = teacher_logits.detach()
        student_probs = (student_logits / self.student_temp).softmax(dim = -1)
        teacher_probs = ((teacher_logits - self.teacher_centers) / self.teacher_temp).softmax(dim = -1)
        loss = - (teacher_probs * torch.log(student_probs + eps)).sum(dim = -1).mean()
        return loss

    def update_moving_average(self, decay=None):
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder, decay=decay)

        new_teacher_centers = self.teacher_centering_ema_updater.update_average(self.teacher_centers, self.previous_centers)
        self.teacher_centers.copy_(new_teacher_centers)

    def forward(self, node_feat1, node_feat2, adj1, adj2, lapl1, lapl2):
        batch_size = node_feat1.shape[0]

        # Concatenate the two views to compute embeddings as one batch.
        node_feat = torch.cat([node_feat1, node_feat2], dim=0)
        adj = torch.cat([adj1, adj2], dim=0)
        lapl = torch.cat([lapl1, lapl2], dim=0)

        _, student_proj = self.student_encoder(node_feat, adj, lapl)
        student_proj1, student_proj2 = torch.split(student_proj, batch_size, dim=0)

        with torch.no_grad():
            _, teacher_proj = self.teacher_encoder(node_feat, adj, lapl)
            teacher_proj1, teacher_proj2 = torch.split(teacher_proj, batch_size, dim=0)

        teacher_logits_avg = teacher_proj.mean(dim = 0)
        self.previous_centers.copy_(teacher_logits_avg)

        loss1 = self.compute_loss(teacher_proj1, student_proj2)
        loss2 = self.compute_loss(teacher_proj2, student_proj1)
        loss = (loss1 + loss2) / 2

        return loss


def create_model(config):
    num_classes = config['model']['num_classes']

    # Create encoder.
    transformer = GraphTransformer(n_nodes=config['data']['n_nodes'],
                 dim=config['model']['dim'], 
                 depth=config['model']['depth'], 
                 num_heads=config['model']['n_head'],
                 feat_dim=config['data']['feat_dim'],
                 pos_dim=config['model']['pos_dim'],
                 num_classes=num_classes)

    # Create GraphDINO.
    model = GraphDINO(transformer,
                 num_classes=num_classes, 
                 moving_average_decay=config['model']['move_avg'],
                 center_moving_average_decay=config['model']['center_avg'],
                 teacher_temp=config['model']['teacher_temp']
                )
    
    return model
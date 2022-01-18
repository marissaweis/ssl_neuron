# Attention and Block adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# DINO adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/dino.py

import copy
import torch
import torch.nn as nn

class GraphAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        inner_dim = dim * num_heads
        self.num_heads = num_heads
        head_dim = inner_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(inner_dim, dim)
        
        self.predict_gamma = nn.Linear(dim, 2)

    def forward(self, x, adj):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # compute bias towards graph adjacency metric
        gamma = self.predict_gamma(x)[:, None].repeat(1, self.num_heads, 1, 1)
        # add column/row to adjacency matrix for class token and repeat per attention head
        bias = torch.eye(N, N)[None,None].repeat(B, self.num_heads, 1, 1).to(x.device)
        bias[:, :, 1:, 1:] = adj[:,None].repeat(1, self.num_heads, 1, 1)
        # weighted sum of transformer attention and adjacency matrix
        attn = gamma[:, :, :, 0:1] * attn + gamma[:, :, :, 1:2] * bias
        
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        return x
    
    
class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GraphAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim=dim, hidden_dim=mlp_hidden_dim)

    def forward(self, x, a):
        x = self.norm1(x)
        x = self.attn(x, a) + x
        x = self.norm2(x)
        x = self.mlp(x) + x
        return x
    
    
class GraphTransformer(nn.Module):
    def __init__(self, n_nodes=200, 
                 dim=32, 
                 depth=5, 
                 heads=8, 
                 mlp_ratio=2., 
                 feat_dim=8,
                 num_classes=1000, 
                 pos_dim=32,
                 proj_dim=128):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, dim))

        self.blocks = nn.Sequential(*[
            Block(dim=dim, num_heads=heads, mlp_ratio=mlp_ratio)
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
        
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(feat_dim, dim*2),
            nn.ReLU(True),
            nn.Linear(dim*2, dim)
        )

    def forward(self, graph, adj, lapl):
        x = self.to_patch_embedding(graph)
        b, n, _ = x.shape

        pos_embedding_token = self.to_pos_embedding(lapl)
        cls_pos_enc = self.cls_pos_embedding.expand(x.shape[0], -1, -1)
        pos_embedding = torch.cat((cls_pos_enc, pos_embedding_token), dim=1)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x += pos_embedding
        for block in self.blocks:
            x = block(x, adj)
        x = x[:, 0]
        x = self.mlp_head(x)

        return x, self.projector(x)



# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new, gamma=None):
        if old is None:
            return new
        if gamma is not None:
            return old * gamma + (1 - gamma) * new
        else:
            return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model, gamma=None):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight, gamma=None)

class GraphDino(nn.Module):
    def __init__(
        self,
        transformer,
        num_classes=1000,
        student_temp=0.9,
        teacher_temp=0.06,
        moving_average_decay=0.999,
        center_moving_average_decay=0.9,
        pos_dim=32,
        n_feat=8,
        n_nodes=200
    ):
        super().__init__()

        self.student_encoder = transformer
        self.teacher_encoder = copy.deepcopy(self.student_encoder)

        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

        self.teacher_ema_updater = EMA(moving_average_decay)

        self.register_buffer('teacher_centers', torch.zeros(1, num_classes))
        self.register_buffer('last_teacher_centers',  torch.zeros(1, num_classes))

        self.teacher_centering_ema_updater = EMA(center_moving_average_decay)

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    def compute_loss(self, teacher_logits, student_logits, eps = 1e-20):
        teacher_logits = teacher_logits.detach()
        student_probs = (student_logits / self.student_temp).softmax(dim = -1)
        teacher_probs = ((teacher_logits - self.teacher_centers) / self.teacher_temp).softmax(dim = -1)
        loss = - (teacher_probs * torch.log(student_probs + eps)).sum(dim = -1).mean()
        return loss

    def update_moving_average(self, gamma=None):
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder, gamma=None)

        new_teacher_centers = self.teacher_centering_ema_updater.update_average(self.teacher_centers, self.last_teacher_centers)
        self.teacher_centers.copy_(new_teacher_centers)

    def forward(self, x1, x2, a1, a2, l1, l2):

        _, student_proj_one = self.student_encoder(x1, a1, l1)
        _, student_proj_two = self.student_encoder(x2, a2, l2)

        with torch.no_grad():
            _, teacher_proj_one = self.teacher_encoder(x1, a1, l1)
            _, teacher_proj_two = self.teacher_encoder(x2, a2, l2)

        teacher_logits_avg = torch.cat((teacher_proj_one, teacher_proj_two)).mean(dim = 0)
        self.last_teacher_centers.copy_(teacher_logits_avg)

        loss1 = self.compute_loss(teacher_proj_one, student_proj_two)
        loss2 = self.compute_loss(teacher_proj_two, student_proj_one)
        loss = (loss1 + loss2) / 2

        return loss
    
    
def create_model(config):
    num_classes = config['model']['num_classes']
    n_nodes = config['data']['n_nodes']
    pos_dim = config['model']['pos_dim']
    teacher_temp = config['model']['teacher_temp']

    
    transformer = GraphTransformer(n_nodes=n_nodes, 
                 dim=config['model']['dim'], 
                 depth=config['model']['depth'], 
                 heads=config['model']['n_head'],
                 feat_dim=config['data']['feat_dim'],
                 pos_dim=pos_dim,
                 num_classes=num_classes)
    
    model = GraphDino(transformer, 
                 num_classes=num_classes, 
                 n_nodes=n_nodes, 
                 pos_dim=pos_dim, 
                 n_feat=config['data']['feat_dim'],
                 moving_average_decay=config['model']['move_avg'],
                 center_moving_average_decay=config['model']['center_avg'],
                 teacher_temp=teacher_temp
                )
    
    return model
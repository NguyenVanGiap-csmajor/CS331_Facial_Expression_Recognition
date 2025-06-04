"""
Complete implementation of the FERPlus model following the MSAD pipeline:
  1) Stem CNN (IR-50) → feature map [B, 512, 14, 14]
  2) Local CNNs (LANet branches) + Multi-Attention Dropping (MAD) → X_out [B, 512, 14, 14]
  3) Projection + Tokenization → token sequence [B, N+1, D]
  4) Transformer Encoder with MSAD blocks (Multi‐head Self‐Attention + MAD + MLP) → sequence [B, N+1, D]
  5) MLP head on [class] token → logits [B, num_classes]

This script contains:
  - IR‐50 backbone definition (from model_irse.py)
  - StemCNN class to extract feature‐map
  - LANetBranch for one attention branch
  - MAD (Multi‐Attention Dropping) for local and per‐head dropping
  - LocalCNN to combine LANet branches + MAD + fuse
  - TransformerEncoderBlock with MSAD inside
  - TransFER class (projection → tokens → MSAD Transformer → head)
  - FullModel that composes StemCNN, LocalCNN, and TransFER

You can import and use `FullModel` in your training loop.
"""

# 1) IR‐50 Backbone Definition (model_irse.py contents)
# =========================
import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple


# Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512),
                                           BatchNorm1d(512))
        else:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512),
                                           BatchNorm1d(512))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self._initialize_weights()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


def IR_50(input_size):
    """Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, 'ir')

    return model


def IR_101(input_size):
    """Constructs a ir-101 model.
    """
    model = Backbone(input_size, 100, 'ir')

    return model


def IR_152(input_size):
    """Constructs a ir-152 model.
    """
    model = Backbone(input_size, 152, 'ir')

    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model.
    """
    model = Backbone(input_size, 50, 'ir_se')

    return model


def IR_SE_101(input_size):
    """Constructs a ir_se-101 model.
    """
    model = Backbone(input_size, 100, 'ir_se')

    return model


def IR_SE_152(input_size):
    """Constructs a ir_se-152 model.
    """
    model = Backbone(input_size, 152, 'ir_se')

    return model

# 2) StemCNN: Extract spatial feature map [B, 512, 14, 14]
# ====================
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class StemCNN(nn.Module):
    """
    - Uses IR_50 backbone to extract a feature map.
    - We ignore the Backbone.output_layer; instead:
      * Run input through input_layer + body ⇒ [B,512,7,7]
      * Then upsample to [B,512,14,14] so that LocalCNN receives H=W=14.
    """
    def __init__(self, pretrained_path, device='cuda'):
        super(StemCNN, self).__init__()
        # 1) Create IR-50 backbone
        self.backbone = IR_50([112, 112])

        # 2) Load pretrained weights (strict=False to skip missing keys if any)
        state_dict = torch.load(pretrained_path, map_location=device)
        if any(k.startswith('module.') for k in state_dict):
            # Remove "module." prefix if saved from DataParallel
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.backbone.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained IR_50 from {pretrained_path}")

    def forward(self, x):
        """
        Input: x ∈ [B, 3, 112, 112]
        Process:
          1) input_layer → [B, 64, 112, 112]
          2) body (Bottleneck blocks) → [B, 512, 7, 7]
          3) Upsample to [B, 512, 14, 14]
        """
        x = self.backbone.input_layer(x)   # [B,64,112,112]
        x = self.backbone.body(x)          # [B,512,7,7]
        x = F.interpolate(x, size=(14, 14), mode='bilinear', align_corners=False)
        # Output: [B,512,14,14]
        return x

# 3) LANet Branch: one 1×1 Conv → ReLU → 1×1 Conv → Sigmoid
# ==================== 
class LANetBranch(nn.Module):
    """
    Creates a single LANet branch:
      - 1×1 Conv from in_channels → reduced_channels
      - ReLU activation
      - 1×1 Conv from reduced_channels → 1
      - Sigmoid to output attention map in [0,1], shape [B, 1, H, W]
    `reduction_ratio` controls how many channels we reduce to.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(LANetBranch, self).__init__()
        # Ensure reduced ≥ 8 (to avoid too small)
        reduced = max(in_channels // reduction_ratio, 8)
        self.conv1 = nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(reduced, 1, kernel_size=1, bias=False)

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns: attention map [B, 1, H, W] in (0,1)
        """
        x = F.relu(self.conv1(x), inplace=True)
        return torch.sigmoid(self.conv2(x))

# 4) MAD (Multi‐Attention Dropping) for local & heads        
# ==========================
class MAD(nn.Module):
    """
    Multi‐Attention Dropping:
    - Given a list of `num_branches` tensors of identical shape,
      with probability p_drop, randomly pick one index k and zero out that entire tensor.
    - If not training or p_drop==0, simply return the list unchanged.
    """
    def __init__(self, num_branches, p_drop=0.5):
        super(MAD, self).__init__()
        self.num_branches = num_branches
        self.p_drop = p_drop

    def forward(self, branch_list):
        """
        branch_list: Python list of length num_branches, each element is [B, ..., ...].
        Returns: same list (possibly with one element zeroed).
        """
        # Only drop during training
        if not self.training or self.p_drop == 0 or random.random() >= self.p_drop:
            return branch_list

        # Choose a random branch to zero
        drop_idx = random.randrange(self.num_branches)
        branch_list[drop_idx] = torch.zeros_like(branch_list[drop_idx])
        return branch_list

# 5) LocalCNN: combine multiple LANetBranch + MAD + fuse
# ========================
class LocalCNN(nn.Module):
    """
    Local CNN module (blue box in diagram):
      1) Receive feature map X [B, C, H, W]
      2) Run through num_branches of LANetBranch ⇒ each returns [B, 1, H, W]
      3) Apply MAD: randomly zero one branch map with prob p_drop
      4) Stack maps → [B, num_branches, 1, H, W]
      5) Element-wise MAX across dim=1 → M_out [B, 1, H, W]
      6) Hadamard: X * M_out → X_w [B, C, H, W]
      7) Optionally Dropout2d on X_w
      8) 1×1 Conv fuse: [B, C, H, W] → [B, C, H, W]
      9) Return fused_map, list_of_att_maps, M_out
    """
    def __init__(self, in_channels, num_branches=4, reduction_ratio=16,
                 p_drop=0.5, dropout_local=0.2):
        super(LocalCNN, self).__init__()
        # Create multiple LANetBranch modules
        self.branches = nn.ModuleList([
            LANetBranch(in_channels, reduction_ratio) for _ in range(num_branches)
        ])
        # MAD to drop one branch at random
        self.mad = MAD(num_branches, p_drop)
        # 1×1 Conv to fuse (keep same in_channels)
        self.fuse = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        # Optional spatial Dropout
        self.dropout = nn.Dropout2d(dropout_local)

    def forward(self, x):
        """
        x: [B, C, H, W]  (from StemCNN)
        returns:
          - fused: [B, C, H, W]  (X_out)
          - att_maps: list of length num_branches, each [B, 1, H, W]
          - M_out: [B, 1, H, W]  (element-wise max across branches)
        """
        # 1) Get attention maps from each branch
        att_maps = [branch(x) for branch in self.branches]  
        # 2) Apply MAD to the list
        att_maps = self.mad(att_maps)                       
        # 3) Stack along new dimension: [B, num_branches, 1, H, W]
        stacked = torch.stack(att_maps, dim=1)               
        # 4) Element-wise max across dimension=1 → [B, 1, H, W]
        M_out, _ = torch.max(stacked, dim=1)                 
        # 5) Hadamard product: [B,C,H,W] * [B,1,H,W] → broadcast to [B,C,H,W]
        X_w = x * M_out                                       
        # 6) Dropout2d on weighted features
        X_w = self.dropout(X_w)                                
        # 7) Fuse via 1×1 convolution
        fused = self.fuse(X_w)                                
        return fused, att_maps, M_out

# 6) Transformer Encoder Block with MSAD (Multi-head + MAD)
# =========================================
class TransformerEncoderBlock(nn.Module):
    """
    One Transformer encoder block with Multi-head Self-Attention Dropping (MSAD):
      a) Input x [B, N, D]
      b) LayerNorm → x_norm
      c) MSA: attn(x_norm, x_norm, x_norm) → ao [B, N, D]
      d) Split ao into 'num_heads' parts of size head_dim = D//num_heads
      e) MAD: randomly zero one head (part) with prob p2
      f) Concat parts → attn_out [B, N, D]
      g) Skip + Dropout: x1 = x + Dropout(attn_out)
      h) LayerNorm on x1 → x2
      i) MLP (2 FC + GELU + Dropout) on x2 → mlp_out
      j) Skip: x_out = x1 + mlp_out
    """
    def __init__(self, dim, num_heads, mlp_ratio=4, p2=0.5, dropout=0.2):
        super(TransformerEncoderBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # (b) LayerNorm before attention
        self.norm1 = nn.LayerNorm(dim)
        # (c) Multi-head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        # (d+e) MSAD for splitting heads and dropping one randomly
        self.msad = MAD(num_heads, p2)
        self.dropout1 = nn.Dropout(dropout)

        # (h) LayerNorm before MLP
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        # (i) MLP: FC → GELU → Dropout → FC → Dropout
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x: [B, N, D]
        return: [B, N, D]
        """
        # (b) LayerNorm
        x_norm = self.norm1(x)                              # [B, N, D]
        # (c) Multi-head attention
        ao, _ = self.attn(x_norm, x_norm, x_norm)            # ao: [B, N, D]

        # (d) Split ao into list of `num_heads` tensors each [B, N, head_dim]
        B, N, D = ao.shape
        # head_dim = D // num_heads
        head_dim = self.head_dim
        parts = [ao[..., i*head_dim:(i+1)*head_dim] for i in range(self.num_heads)]
        # (e) MAD: randomly zero one head with prob p2
        parts = self.msad(parts)  # returns same-length list

        # (f) Concat back along last dim → [B, N, D]
        attn_out = torch.cat(parts, dim=-1)
        # (g) Skip connection + dropout on attention output
        x1 = x + self.dropout1(attn_out)                     # [B, N, D]

        # (h) LayerNorm before MLP
        x2 = self.norm2(x1)                                  # [B, N, D]
        # (i) MLP
        mlp_out = self.mlp(x2)                               # [B, N, D]
        # (j) Skip connection
        x_out = x1 + mlp_out                                 # [B, N, D]
        return x_out

# 7) TransFER: Projection → Tokenization → MSAD Transformer → Head
# ==============================
class TransFER(nn.Module):
    """
    TransFER module (right‐blue box in diagram):
      1) Input: X_out [B, C, H, W] from LocalCNN
      2) Projection: 1×1 Conv to D channels → [B, D, H, W]
      3) Flatten spatial dims → [B, H*W, D]
      4) Prepend learnable [class] token → [B, H*W+1, D]
      5) Add learnable positional embedding [1, H*W+1, D]
      6) Apply dropout on token embeddings
      7) Pass through stack of `depth` TransformerEncoderBlock (MSAD inside)
      8) LayerNorm final
      9) MLP head (Linear) on token[0] → logits [B, num_classes]
    """
    def __init__(self, in_channels, proj_channels, num_patches, num_classes,
                 depth=8, num_heads=8, mlp_ratio=4, p2=0.5, dropout=0.2):
        super(TransFER, self).__init__()

        # (2) 1×1 Conv: from in_channels (e.g. 512) → proj_channels (embedding dim)
        self.proj = nn.Conv2d(in_channels, proj_channels, kernel_size=1, bias=False)
        # (6) Dropout on token embeddings
        self.dropout_in = nn.Dropout(dropout)

        # (4) Learnable [class] token param, shape [1, 1, proj_channels]
        self.cls = nn.Parameter(torch.zeros(1, 1, proj_channels))
        # (5) Learnable positional embeddings, shape [1, num_patches+1, proj_channels]
        self.pos = nn.Parameter(torch.zeros(1, num_patches + 1, proj_channels))

        # (7) Stack of `depth` TransformerEncoderBlock with MSAD
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(proj_channels, num_heads, mlp_ratio, p2, dropout)
            for _ in range(depth)
        ])

        # (8) Final LayerNorm on sequence
        self.norm = nn.LayerNorm(proj_channels)
        # (9) Classification head: Linear from proj_channels → num_classes
        self.head = nn.Linear(proj_channels, num_classes)

        # Initialize CLS token & positional embedding (Truncated normal)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        """
        x: [B, C, H, W] (LocalCNN output), here C=512, H=W=14
        Returns: logits [B, num_classes]
        """
        B, C, H, W = x.shape
        # (2) 1×1 Conv → [B, proj_channels, H, W]
        x = self.proj(x)

        # (3) Flatten spatial dims: [B, proj_channels, H*W] → transpose → [B, H*W, proj_channels]
        x = x.flatten(2).transpose(1, 2)

        # (4) Expand cls token: [1,1,D] → [B,1,D]
        cls_tokens = self.cls.expand(B, -1, -1)  # [B,1,D]
        # (5) Prepend cls & add positional: [B, H*W+1, D]
        x = torch.cat((cls_tokens, x), dim=1) + self.pos

        # (6) Dropout on initial token embeddings
        x = self.dropout_in(x)

        # (7) Pass through Transformer blocks (MSAD inside)
        for block in self.blocks:
            x = block(x)

        # (8) Final LayerNorm
        x = self.norm(x)

        # (9) Take [class] token (index 0) and apply MLP head → logits
        cls_out = x[:, 0, :]            # [B, D]
        logits = self.head(cls_out)     # [B, num_classes]
        return logits

# 8) FullModel: compose StemCNN → LocalCNN → TransFER
# ============================
class FullModel(nn.Module):
    """
    Full pipeline:
      1) StemCNN: IR-50 → feature map [B,512,14,14]
      2) LocalCNN: LANet branches + MAD → X_out [B,512,14,14]
      3) TransFER: projection → MSAD Transformer → MLP head → logits
    """
    def __init__(self, stem: StemCNN, local: LocalCNN, transfer: TransFER):
        super(FullModel, self).__init__()
        self.stem = stem        # Stem CNN (IR-50)
        self.local = local      # Local CNNs + MAD
        self.transfer = transfer  # TransFER module

    def forward(self, x):
        # (1) Stem CNN
        feats = self.stem(x)            # [B,512,14,14]
        # (2) Local CNN + MAD
        fused, att_maps, M_out = self.local(feats)  # fused: [B,512,14,14]
        # (3) TransFER → logits
        logits = self.transfer(fused)   # [B, num_classes]
        return logits
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import math


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, LayerNorm) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.weight + self.bias


class MHSA(nn.Module):
    """ Multi-Head Self-Attention

    """
    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=False, attn_drop=0., proj_drop=0., norm=None, size=None, pos_encoding=False):
        super().__init__()
        assert not (pos_encoding and size is None), 'When using positional encoding, size can not be None'
        norm = norm or nn.Identity

        self.num_heads = num_heads
        self.pos_encoding = pos_encoding
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.query = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.key = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.value = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = norm(dim)

        if pos_encoding:
            width, height = size
            self.rel_h = nn.Parameter(torch.randn([1, num_heads, head_dim, 1, height // sr_ratio]))
            self.rel_w = nn.Parameter(torch.randn([1, num_heads, head_dim, width // sr_ratio, 1]))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, self.num_heads, C // self.num_heads, -1)
        if self.sr_ratio > 1:
            x = self.norm(self.sr(x))
        k = self.key(x).view(B, self.num_heads, C // self.num_heads, -1)
        v = self.value(x).view(B, self.num_heads, C // self.num_heads, -1)
        if self.pos_encoding:
            pos = (self.rel_h + self.rel_w).view(1, self.num_heads, C // self.num_heads, -1)
            k = k + pos
        k = k * self.scale
        attn = q.transpose(-2, -1) @ k
        attn = self.attn_drop(self.softmax(attn))
        out = (v @ attn.transpose(-2, -1)).view(B, C, H, W)
        out = self.proj_drop(self.proj(out))

        return out


class IRFFN(nn.Module):
    """ Inverted Residual Feed Forward Network

    """
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv2(x)
        x = self.conv3(x)

        return x


class CMTL(nn.Module):
    """ Convolution Meet Transformer Layer

    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4, sr_ratio=1, qkv_bias=False, drop=0., attn_drop=0., drop_path=0., norm=None, size=None, pos_encoding=False):
        super().__init__()
        norm = norm or nn.Identity
        self.norm1 = norm(dim)
        self.norm2 = norm(dim)

        self.attn = MHSA(dim=dim, num_heads=num_heads, sr_ratio=sr_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop,
                         proj_drop=drop, norm=norm, size=size, pos_encoding=pos_encoding)
        self.ffn = IRFFN(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x


class CMTB(nn.Module):
    """ Convolution Meet Transformer Block

    """
    def __init__(self, dim, depth, num_heads=8, mlp_ratio=4, sr_ratio=1, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.1, norm=LayerNorm, size=None, pos_encoding=True, **kwargs):
        super().__init__()
        norm = norm or nn.Identity
        self.norm1 = norm(dim)
        self.norm2 = norm(dim)

        layers = []
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
        for i in range(depth):
            layers.append(
                CMTL(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    sr_ratio=sr_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm=norm,
                    size=size,
                    pos_encoding=pos_encoding
                )
            )
        self.layer = nn.Sequential(*layers)

        self.apply(init_weights)

    def forward(self, x):
        x = self.norm2(self.layer(self.norm1(x)))

        return x


if __name__ == '__main__':
    device = 'cuda:0'
    model = CMTB(dim=256, depth=1, sr_ratio=16, size=(128, 128)).to(device)

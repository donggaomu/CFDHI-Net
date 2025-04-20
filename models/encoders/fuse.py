import torch
import torch.nn as nn

class Channel_Attention(nn.Module):
    def __init__(self, dim, ratio=16):
        super(Channel_Attention, self).__init__()
        self.gmp_pool = nn.AdaptiveMaxPool2d(1)
        self.down = nn.Linear(dim, dim//ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim//ratio, dim)
    def forward(self, x):
        max_out = self.up(self.act(self.down(self.gmp_pool(x).permute(0,2,3,1)))).permute(0,3,1,2)
        return max_out

class Feature_Pool(nn.Module):
    def __init__(self, dim, ratio=2):
        super(Feature_Pool, self).__init__()
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=7, padding=3, groups=dim)
        self.T_c = nn.Parameter(torch.ones([]) * dim)
        self.cse = Channel_Attention(dim * 2)

        self.down = nn.Linear(dim, dim * ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim * ratio, dim)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.up(self.act(self.down((self.gap_pool(x)+self.max_pool(x)).permute(0,2,3,1)))).permute(0,3,1,2).view(b,c)
        return y

class Decouple(nn.Module):
    def __init__(self, dim, reduction=2):
        super(Decouple, self).__init__()
        self.dim = dim

        self.mlp_pool = Feature_Pool(dim, reduction)
        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=7, padding=3, groups=dim)
        self.T_c = nn.Parameter(torch.ones([]) * dim)
        self.cse = Channel_Attention(dim * 2)

    def forward(self, RGB, T):
        b, c, h, w = RGB.shape
        rgb_y = self.mlp_pool(RGB)
        t_y = self.mlp_pool(T)
        rgb_y = rgb_y / rgb_y.norm(dim=1, keepdim=True)
        t_y = t_y / t_y.norm(dim=1, keepdim=True)
        rgb_y = rgb_y.view(b, c, 1)
        t_y = t_y.view(b, 1, c)
        logits_per = self.T_c * rgb_y @ t_y
        cross_gate = torch.diagonal(torch.sigmoid(logits_per)).reshape(b, c, 1, 1)
        add_gate = torch.ones(cross_gate.shape).cuda() - cross_gate

        New_RGB_I = RGB * cross_gate
        New_T_I = T * cross_gate
        x_cat = torch.cat((New_RGB_I, New_T_I), dim=1)
        ##########################################################################
        fuse_gate = torch.sigmoid(self.cse(self.dwconv(x_cat)))
        rgb_gate, t_gate = fuse_gate[:, 0:c, :], fuse_gate[:, c:c * 2, :]
        ##########################################################################
        New_RGB_I = RGB * rgb_gate
        New_T_I = T * t_gate

        New_RGB_C = RGB * add_gate
        New_T_C = T * add_gate
        return New_RGB_I, New_RGB_C, New_T_I, New_T_C



class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        # B, N, _C = x.shape
        # x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out

class Mcsff(nn.Module):
    def __init__(self, dim, reduction=4):
        super(Mcsff, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim // reduction, self.dim),
            nn.Sigmoid())
        self.mlp2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid())
        self.sim_channel = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.sim_spa = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.fuse = ChannelEmbed(2*dim, dim)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1_avg = self.avg_pool(x1).view(B, self.dim, -1)
        x1_max = self.max_pool(x1).view(B, self.dim, -1)
        x1_pool = x1_avg+x1_max

        x2_avg = self.avg_pool(x2).view(B, self.dim, -1)
        x2_max = self.max_pool(x2).view(B, self.dim, -1)
        x2_pool = x2_avg + x2_max

        distance = self.sim_channel(x1_pool, x2_pool)
        w = self.mlp(distance)

        x1_c = w[..., None, None]*x1
        x2_c = w[..., None, None]*x2

        x1_c_avg = torch.mean(x1_c, dim=1, keepdim=True)
        x1_c_max, _ = torch.max(x1_c, dim=1, keepdim=True)
        x1_c_pool = x1_c_max + x1_c_avg

        x2_c_avg = torch.mean(x2_c, dim=1, keepdim=True)
        x2_c_max, _ = torch.max(x2_c, dim=1, keepdim=True)
        x2_c_pool = x2_c_max + x2_c_avg

        dis = self.sim_spa(x1_c_pool, x2_c_pool)
        w2 = self.mlp2(dis.unsqueeze(1))

        x1_out = x1_c*w2 + x1
        x2_out = x2_c*(1-w2) + x2

        fuse = x1_out+x2_out
        # out = self.fuse(fuse, H, W)
        return fuse

class Mdff(nn.Module):
    def __init__(self, inc, ratio=4):
        super(Mdff, self).__init__()
        self.fc_g = nn.Sequential(*[
            nn.Linear(inc, inc // ratio, False),
            nn.ReLU(),
            nn.Linear(inc // ratio, inc, False),
            nn.Sigmoid()
        ])

        self.conv = nn.Sequential(*[
            nn.Conv2d(2*inc, inc, 3, 1, 1),
            nn.BatchNorm2d(inc),
            nn.ReLU(),
        ])

        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_g1, x_g2):

        B, _, _, _ = x_g1.shape
        x_g = x_g1-x_g2
        # x_d = x_d1+x_d2

        x_g_m = self.gmp(x_g).view(B, -1)
        x_g_a = self.gap(x_g).view(B, -1)

        g_weights = self.fc_g(x_g_m+x_g_a)
        # d_weights = self.fc_d(x_d_a)

        x_g1_w = x_g1*g_weights[:, :, None, None]
        x_g2_w = x_g2*g_weights[:, :, None, None]
        # x_g_w = torch.cat([x_g1_w, x_g2_w], dim=1)
        # x_g_w = x_g1_w+x_g2_w
        x_g1_new = x_g1+x_g1_w
        x_g2_new = x_g2+x_g2_w
        fuse = self.conv(torch.cat([x_g1_new, x_g2_new], dim=1))

        return  fuse

class Fusion(nn.Module):
    def __init__(self, dim):
        super(Fusion, self).__init__()
        self.decouple = Decouple(dim)
        self.i_fuse = Mcsff(dim)
        self.c_fuse = Mdff(dim)
        self.fuse = ChannelEmbed(2*dim, dim)
    
    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1_i, x1_c, x2_i, x2_c = self.decouple(x1, x2)
        fuse_i = self.i_fuse(x1_i, x2_i)
        fuse_c = self.c_fuse(x1_c, x2_c)
        fuse = self.fuse(torch.cat([fuse_i, fuse_c], dim=1), H, W)
        return fuse


if __name__ == '__main__':
    input1 = torch.randn(4, 256, 120, 160)
    input2 = torch.randn(4, 256, 120, 160)

    net = Fusion(dim=256)
    out = net(input1, input2)



from torchvision.models._utils import IntermediateLayerGetter
import torch
import torch.nn as nn
import torch.nn.functional as F
from geoseg.models.backbones import (resnet)
import math



def conv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6(inplace=True)
    )


class GCG(nn.Module):
    def __init__(self):
        super(GCG, self).__init__()

    def forward(self, x, preds):
        inputs = x
        batch_size, num_channels, h, w = x.size()
        num_classes = preds.size(1)

        feats_semantic = torch.zeros(batch_size, h*w, num_channels).type_as(x)

        for batch_idx in range(batch_size):
            feats_iter, preds_iter = x[batch_idx], preds[batch_idx]
            feats_iter, preds_iter = feats_iter.reshape(num_channels, -1), preds_iter.reshape(num_classes, -1)
            feats_iter, preds_iter = feats_iter.permute(1, 0), preds_iter.permute(1, 0)
            argmax = preds_iter.argmax(1)

            for clsid in range(num_classes):
                mask = (argmax == clsid)
                if mask.sum() == 0: continue
                feats_iter_cls = feats_iter[mask]
                preds_iter_cls = preds_iter[:, clsid][mask]
                weight = F.softmax(preds_iter_cls, dim=0)
                feats_iter_cls = feats_iter_cls * weight.unsqueeze(-1)
                # print(feats_iter_cls.shape)
                feats_iter_cls = feats_iter_cls.sum(0)
                feats_semantic[batch_idx][mask] = feats_iter_cls

        feats_semantic = feats_semantic.reshape(batch_size, h, w, num_channels).permute(0, 3, 1, 2).contiguous()
        # print(feats_semantic.shape)

        return feats_semantic





class GlobalClassGatherModule(nn.Module):
    def __init__(self, input_channels=128):
        super(GlobalClassGatherModule, self).__init__()
        self.input_channels = input_channels
        self.gcc = GCG()
        self.conv1 = nn.Conv2d(input_channels, input_channels, 1, 1)
        self.conv2 = nn.Conv2d(input_channels, input_channels, 1, 1)


    def forward(self, x, preds):
        b, c, h, w = x.size()
        feat_semantic = self.gcc(x, preds)
        x = self.conv1(x)
        x = x.reshape(b, c, -1).contiguous()
        feat_semantic = self.conv2(feat_semantic)
        feat_semantic = feat_semantic.reshape(b, c, -1).permute(0, 2, 1).contiguous()
        gc = torch.matmul(x, feat_semantic)
        gc = (self.input_channels ** -0.5) * gc
        gc = F.softmax(gc, dim=-1)

        return gc


def patch_split(input, patch_size):
    """
    input: (B, C, H, W)
    output: (B*num_h*num_w, C, patch_h, patch_w)
    """
    B, C, H, W = input.size()
    num_h, num_w = patch_size
    patch_h, patch_w = H // num_h, W // num_w
    out = input.view(B, C, num_h, patch_h, num_w, patch_w)
    out = out.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, patch_h,
                                                          patch_w)  # (B*num_h*num_w, C, patch_h, patch_w)
    return out


def patch_recover(input, patch_size):
    """
    input: (B*num_h*num_w, C, patch_h, patch_w)
    output: (B, C, H, W)
    """
    N, C, patch_h, patch_w = input.size()
    num_h, num_w = patch_size
    H, W = num_h * patch_h, num_w * patch_w
    B = N // (num_h * num_w)

    out = input.view(B, num_h, num_w, C, patch_h, patch_w)
    out = out.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    return out



class SelfAttentionBlock(nn.Module):
    """
    query_feats: (B*num_h*num_w, C, patch_h, patch_w)
    key_feats: (B*num_h*num_w, C, K, 1)
    value_feats: (B*num_h*num_w, C, K, 1)

    output: (B*num_h*num_w, C, patch_h, patch_w)
    """

    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels,
                 key_query_num_convs, value_out_num_convs):
        super(SelfAttentionBlock, self).__init__()
        self.key_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs,
        )
        self.query_project = self.buildproject(
            in_channels=query_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs
        )
        self.value_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=value_out_num_convs
        )
        self.out_project = self.buildproject(
            in_channels=transform_channels,
            out_channels=out_channels,
            num_convs=value_out_num_convs
        )
        self.transform_channels = transform_channels

    def forward(self, query_feats, key_feats, value_feats):
        batch_size = query_feats.size(0)

        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()  # (B*num_h*num_w, patch_h*patch_w, C)
        print(query.shape)

        key = self.key_project(key_feats)
        key = key.reshape(*key.shape[:2], -1)  # (B*num_h*num_w, C, patch_h*patch_w)

        value = self.value_project(value_feats)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()  # (B*num_h*num_w, patch_h*patch_w, C)

        sim_map = torch.matmul(query, key)

        sim_map = (self.transform_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)  # (B*num_h*num_w, patch_h*patch_w, patch_h*patch_w)


        context = torch.matmul(sim_map, value)  # (B*num_h*num_w, patch_h*patch_w, C)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])  # (B*num_h*num_w, C, patch_h, patch_w)

        context = self.out_project(context)  # (B*num_h*num_w, C, patch_h, patch_w)
        return context

    def buildproject(self, in_channels, out_channels, num_convs):
        convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        for _ in range(num_convs - 1):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        if len(convs) > 1:
            return nn.Sequential(*convs)
        return convs[0]


class LocalClassGatherModule(nn.Module):
    def __init__(self, scale=1):
        super(LocalClassGatherModule, self).__init__()
        self.scale = scale

    def forward(self, features, probs):
        batch_size, num_classes, h, w = probs.size()
        probs = probs.view(batch_size, num_classes, -1)  # batch * k * hw
        probs = F.softmax(self.scale * probs, dim=2)

        features = features.view(batch_size, features.size(1), -1)
        features = features.permute(0, 2, 1)  # batch * hw * c

        ocr_context = torch.matmul(probs, features)  # (B, k, c)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(-1)  # (B, C, K, 1)
        return ocr_context

def Recover(feats, preds, context):
    B, C, H, W = feats.size()
    preds = torch.argmax(preds, dim=1) #[B, H, W]
    context = context.squeeze(-1).permute(0, 2, 1).contiguous() #[B, K, C]
    new_context = torch.zeros(B, H*W, C).type_as(feats)

    for batch_idx in range(B):
        context_iter, preds_iter = context[batch_idx], preds[batch_idx] # [K, C] [H, W]
        preds_iter = preds_iter.view(-1)  # [HW]
        new_context[batch_idx] = context_iter[preds_iter]

    new_context = new_context.permute(0, 2, 1).view(B, C, H, W)
    return new_context


class GLCA(nn.Module):
    """
    feat: (B, C, H, W)
    global_center: (B, C, K, 1)
    """

    def __init__(self, in_channels, inner_channels, num_class, patch_size=(8, 8)):
        super(GLCA, self).__init__()
        self.num_class = num_class
        self.patch_size = patch_size
        self.project = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.feat_decoder = nn.Conv2d(in_channels, num_class, kernel_size=1)
        self.relu = nn.ReLU6()

        self.correlate_net = SelfAttentionBlock(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            transform_channels=inner_channels,
            out_channels=in_channels,
            key_query_num_convs=2,
            value_out_num_convs=1
        )

        self.long_net = SelfAttentionBlock(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            transform_channels=inner_channels,
            out_channels=in_channels,
            key_query_num_convs=2,
            value_out_num_convs=1
        )

        self.get_center = LocalClassGatherModule()

        self.cat_conv = nn.Sequential(
            conv_3x3(in_channels * 2, in_channels),
            nn.Dropout2d(0.1),
            conv_3x3(in_channels, in_channels),
            nn.Dropout2d(0.1)
        )

    def forward(self, feat, global_center):
        b, c, h, w = feat.size()
        pred = self.feat_decoder(feat)
        residual = feat
        feat = self.project(feat)
        feat = feat.reshape(b, c, -1).contiguous()
        feat = torch.matmul(global_center, feat)
        feat = feat.reshape(b, c, h, w).contiguous()
        feat = residual + feat
        feat = self.relu(feat)


        num_h, num_w = self.patch_size

        patch_h, patch_w = math.ceil(h / num_h), math.ceil(w / num_w)
        pad_h, pad_w = patch_h * num_h - h, patch_w * num_w - w
        if h % num_h != 0 and pad_w > 0:
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            feat = F.pad(feat, padding)
            pred = F.pad(pred, padding)

        patch_feat = patch_split(feat, self.patch_size)
        patch_pred = patch_split(pred, self.patch_size)  # (B*num_h*num_w, K, patch_h, patch_w)
        localclass_center = self.get_center(patch_feat, patch_pred)  # (B*num_h*num_w, C, K)
        local_context = Recover(patch_feat, patch_pred, localclass_center)

        new_feat = self.correlate_net(patch_feat, local_context, local_context)   # (B*num_h*num_w, C, patch_h, patch_w)
        new_feat = patch_recover(new_feat, self.patch_size)

        if pad_h > 0 or pad_w > 0:
            new_feat = new_feat[:, :, pad_h//2: pad_h//2+h, pad_w//2: pad_w//2+w]

        out = self.cat_conv(torch.cat([residual, new_feat], dim=1))
        return out

class MSDC(nn.Module):
    def __init__(self, in_features, hidden_features, dilation=[1, 6, 12]):
        super(MSDC, self).__init__()
        self.fsconv = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.conv1 = nn.Conv2d(hidden_features, hidden_features, 3, 1, padding=1, groups=hidden_features, dilation=dilation[0])
        self.conv2 = nn.Conv2d(hidden_features, hidden_features, 3, 1, padding=6, groups=hidden_features, dilation=dilation[1])
        self.conv3 = nn.Conv2d(hidden_features, hidden_features, 3, 1, padding=12, groups=hidden_features, dilation=dilation[2])

        self.outconv = nn.Conv2d(hidden_features, in_features, 1, 1)

        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.bn2 = nn.BatchNorm2d(hidden_features)
        self.bn3 = nn.BatchNorm2d(in_features)

        self.relu = nn.ReLU6()

    def forward(self, x):
        shortcut = x
        x_1 = self.fsconv(x)
        x_1 = self.bn1(x_1)
        x_3 = self.conv1(x_1)
        x_6 = self.conv2(x_1)
        x_12 = self.conv3(x_1)
        x_ = x_3 + x_6 + x_12
        x_ = self.bn2(x_)
        x_ = self.relu(x_)
        x_ = self.bn3(self.outconv(x_))
        x_ = x_ + shortcut
        x_ = self.relu(x_)
        return x_

class CGTB(nn.Module):
    def __init__(self, transform_channel=128, num_class=6):
        super(CGTB, self).__init__()
        self.center = GLCA(transform_channel, transform_channel // 2, num_class)
        self.mlp = MSDC(128, 64, dilation=[1, 6, 12])

    def forward(self, x, global_center):
        x = self.center(x, global_center)
        x = self.mlp(x)
        return x




def upsample_add(x_small, x_big):
    x_small = F.interpolate(x_small, scale_factor=2, mode="bilinear", align_corners=False)
    return torch.cat([x_small, x_big], dim=1)


class Decode_Head(nn.Module):
    def __init__(self, in_channel=[256, 512, 1024, 2048], transform_channel=128, num_class=6):
        super(Decode_Head, self).__init__()
        self.bottleneck1 = conv_3x3(in_channel[0], transform_channel)
        self.bottleneck2 = conv_3x3(in_channel[1], transform_channel)
        self.bottleneck3 = conv_3x3(in_channel[2], transform_channel)
        self.bottleneck4 = conv_3x3(in_channel[3], transform_channel)

        self.decoder_stage1 = nn.Conv2d(transform_channel, num_class, kernel_size=1)
        self.global_gather = GlobalClassGatherModule()

        self.center1 = CGTB(transform_channel, num_class)
        self.center2 = CGTB(transform_channel, num_class)
        self.center3 = CGTB(transform_channel, num_class)
        self.center4 = CGTB(transform_channel, num_class)

        self.catconv1 = conv_3x3(transform_channel * 2, transform_channel)
        self.catconv2 = conv_3x3(transform_channel * 2, transform_channel)
        self.catconv3 = conv_3x3(transform_channel * 2, transform_channel)

        self.catconv = conv_3x3(transform_channel, transform_channel)

    def forward(self, x1, x2, x3, x4):
        feat1, feat2, feat3, feat4 = self.bottleneck1(x1), self.bottleneck2(x2), self.bottleneck3(x3), self.bottleneck4(
            x4)
        pred1 = self.decoder_stage1(feat4)

        global_center = self.global_gather(feat4, pred1)

        new_feat4 = self.center4(feat4, global_center)

        feat3 = self.catconv1(upsample_add(new_feat4, feat3))
        new_feat3 = self.center3(feat3, global_center)

        feat2 = self.catconv2(upsample_add(new_feat3, feat2))
        new_feat2 = self.center2(feat2, global_center)

        feat1 = self.catconv3(upsample_add(new_feat2, feat1))
        new_feat1 = self.center1(feat1, global_center)

        new_feat4 = F.interpolate(new_feat4, scale_factor=8, mode="bilinear", align_corners=False)
        new_feat3 = F.interpolate(new_feat3, scale_factor=4, mode="bilinear", align_corners=False)
        new_feat2 = F.interpolate(new_feat2, scale_factor=2, mode="bilinear", align_corners=False)

        out = self.catconv(new_feat1 + new_feat2 + new_feat3 + new_feat4)

        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.backbone = resnet.resnet50(pretrained=True)
        self.backbone = IntermediateLayerGetter(self.backbone,
                                                return_layers={'layer1': "res1", "layer2": "res2",
                                                               "layer3": "res3", "layer4": "res4"})
        self.num_classes = num_classes
        self.seghead = Decode_Head(num_class=self.num_classes)
        self.classifier = nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        _, _, h, w = x.size()
        features = self.backbone(x)
        x1 = features['res1']
        x2 = features['res2']
        x3 = features['res3']
        x4 = features['res4']
        out = self.seghead(x1, x2, x3, x4)
        pred = self.classifier(out)
        pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
        return pred


if __name__ == '__main__':
    # from thop import profile
    #
    # model = MyModel(num_classes=6).cuda()
    # x = torch.randn(1, 3, 512, 512).cuda()
    # flops, params = profile(model, (x,))
    # out = model(x)
    # print(out.shape)
    #
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    # net = GLCA(in_channels=128, inner_channels=64, num_class=6).cuda()
    # x = torch.randn(4, 128, 32, 32).cuda()
    # gc = torch.randn(4, 128, 128).cuda()
    # out = net(x, gc)
    # print(out.shape)

    net = GCG().cuda()
    x = torch.randn(3,128, 32, 32).cuda()
    preds = torch.randn(3, 6, 32, 32).cuda()
    out = net(x, preds)


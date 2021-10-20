import torch
import torch.nn as nn
from registration_dense.model_utils import Squeeze, BatchNormPoint
from registration_dense.mv_utils import PCViews
from registration_dense.pspnet import PSPNet

class MVModel(nn.Module):
    def __init__(self, task = 'cls', backbone = 'resnet18',
                 feat_size = 16):

        super().__init__()
        assert task == 'cls'
        self.task = task
        self.output_feature_dim = 1024
        self.dropout_p = 0.5
        self.feat_size = feat_size

        pc_views = PCViews()
        self.num_views = pc_views.num_views
        self._get_img = pc_views.get_img

        img_layers, in_features = self.get_img_layers(
            backbone, feat_size=feat_size)
        self.img_model = Feature2DNet()



    def forward(self, pc):
        """
        :param pc:
        :return:
        """

        pc = pc.cuda()
        img, coordinates = self.get_img(pc)
        #print("img.shape", img.shape)
        #print("coordinates.shape: ", coordinates.shape)
        feat = self.img_model(img, coordinates)
        feat = feat.view((-1, self.num_views*256, 1024))
        #print("feat.shape: ", feat.shape)
        #out = {'logit': logit}
        #return out
        return feat

    def get_img(self, pc):
        img, coordinates = self._get_img(pc)
        img = torch.tensor(img).float()
        #print("non_device", next(self.parameters()))
        #print("device", next(self.parameters()).device)
        #print("img.shape", img.shape)
        #img = img.to(next(self.parameters()).device)
        #print("coordinates.shape: ", coordinates.shape)
        #img.shape torch.Size([192, 128, 128])
        #coordinates.shape:  torch.Size([192, 1024])
        assert len(img.shape) == 3
        img = img.unsqueeze(3)
        # [num_pc * num_views, 1, RESOLUTION, RESOLUTION]
        img = img.permute(0, 3, 1, 2)

        return img, coordinates

    @staticmethod
    def get_img_layers(backbone, feat_size):
        """
        Return layers for the image model
        """

        from registration.resnet import _resnet, BasicBlock
        assert backbone == 'resnet18'
        layers = [2, 2, 2, 2]
        block = BasicBlock
        backbone_mod = _resnet(
            arch=None,
            block=block,
            layers=layers,
            pretrained=False,
            progress=False,
            feature_size=feat_size,
            zero_init_residual=True)

        all_layers = [x for x in backbone_mod.children()]
        in_features = all_layers[-1].in_features

        # all layers except the final fc layer and the initial conv layers
        # WARNING: this is checked only for resnet models
        main_layers = all_layers[4:-1]
        img_layers = [
            nn.Conv2d(1, feat_size, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(feat_size, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            *main_layers,
            Squeeze()
        ]

        return img_layers, in_features


class Feature2DNet(nn.Module):
    def __init__(self):
        super(Feature2DNet, self).__init__()
        self.psp = PSPNet(bins=(1, 2, 3, 6), backend='resnet18')
        self.instance_color = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 256, 1),
            nn.ReLU(),
        )

    def forward(self, img, choose):
        bs = img.size()[0]
        out_img = self.psp(img)
        di = out_img.size()[1]
        emb = out_img.view(bs, di, -1)
        choose = choose.type(torch.cuda.LongTensor)
        choose = choose.unsqueeze(1).repeat(1, di, 1)
        #print("emb0.shape:", emb.shape)
        #print("choose.shape: ", choose.shape)
        emb = torch.gather(emb, 2, choose).contiguous()
        emb = self.instance_color(emb)

        return emb

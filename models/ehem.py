import math  # 导入数学库
import torch  # 导入PyTorch库，这是一个开源的深度学习平台
from torch import nn  # 从torch库中导入nn模块，这个模块包含了各种神经网络的层
import pytorch_lightning as pl  # 导入PyTorch Lightning库，这是一个用于深度学习的轻量级PyTorch封装库

from models.dgcnn import GeoFeatGenerator, GeoFeatGenerator2coord  # 从models.dgcnn模块中导入GeoFeatGenerator和GeoFeatGenerator2coord，这是你自定义的模块，用于生成地理特征
from models.swin_transformer import SwinEncoder, SwinConfig  # 从models.swin_transformer模块中导入SwinEncoder和SwinConfig，这是你自定义的模块，用于实现Swin Transformer模型

class EHEM(pl.LightningModule):  # 定义一个类EHEM，继承自PyTorch Lightning的LightningModule类
    def __init__(self, cfg):  # 定义初始化函数
        super(EHEM, self).__init__()  # 调用父类的初始化函数
        self.cfg = cfg  # 将配置文件保存到self.cfg中

        self.geo_feat_generator = GeoFeatGenerator(max_level=cfg.model.max_level)  # 创建一个GeoFeatGenerator对象

        swin_cfg = SwinConfig(  # 创建一个SwinConfig对象
            num_channels=256,  # 设置通道数为256
            embed_dim=256,  # 设置嵌入维度为256
            depths=[4, 4, 4, 4, 2],  # 设置每一层的深度
            num_heads=[4, 4, 4, 4, 4],  # 设置每一层的头数
            window_size=512,  # 设置窗口大小为512
        )
        self.swin_self_transformer = SwinEncoder(swin_cfg, 8192, False)  # 创建一个SwinEncoder对象
        cross_swin_cfg = SwinConfig(  # 创建一个SwinConfig对象
            num_channels=256,  # 设置通道数为256
            embed_dim=256,  # 设置嵌入维度为256
            depths=[2, 2, 1, 1],  # 设置每一层的深度
            num_heads=[4, 4, 4, 4],  # 设置每一层的头数
            window_size=512,  # 设置窗口大小为512
        )
        self.swin_cross_transformer = SwinEncoder(cross_swin_cfg, 4096, True)  # 创建一个SwinEncoder对象

        self.ancient_mlp = nn.Sequential(  # 创建一个多层感知机
            nn.Linear(1280, 1024),  # 添加一个线性层
            nn.LeakyReLU(),  # 添加一个LeakyReLU激活函数
            nn.Linear(1024, 512),  # 添加一个线性层
            nn.LeakyReLU(),  # 添加一个LeakyReLU激活函数
            nn.Linear(512, 256),  # 添加一个线性层
        )
        self.prob_pred_mlp1 = nn.Sequential(  # 创建一个多层感知机
            nn.Linear(256, 256),  # 添加一个线性层
            nn.LeakyReLU(),  # 添加一个LeakyReLU激活函数
            nn.Linear(256, 256),  # 添加一个线性层
            nn.LeakyReLU(),  # 添加一个LeakyReLU激活函数
            nn.Linear(256, 255),  # 添加一个线性层
        )
        self.pre_occ_mlp = nn.Sequential(  # 创建一个多层感知机
            nn.Linear(16, 16),  # 添加一个线性层
            nn.LeakyReLU(),  # 添加一个LeakyReLU激活函数
            nn.Linear(16, 16),  # 添加一个线性层
            nn.LeakyReLU(),  # 添加一个LeakyReLU激活函数
            nn.Linear(16, 16),  # 添加一个线性层
        )
        self.pre_attn_mlp = nn.Sequential(  # 创建一个多层感知机
            nn.Linear(256, 256),  # 添加一个线性层
            nn.LeakyReLU(),  # 添加一个LeakyReLU激活函数
            nn.Linear(256, 240),  # 添加一个线性层
            nn.LeakyReLU(),  # 添加一个LeakyReLU激活函数
            nn.Linear(240, 240),  # 添加一个线性层
        )
        self.prob_pred_mlp2 = nn.Sequential(  # 创建一个多层感知机
            nn.Linear(1280, 768),  # 添加一个线性层
            nn.LeakyReLU(),  # 添加一个LeakyReLU激活函数
            nn.Linear(768, 512),  # 添加一个线性层
            nn.LeakyReLU(),  # 添加一个LeakyReLU激活函数
            nn.Linear(512, 255),  # 添加一个线性层
        )

        self.criterion = nn.CrossEntropyLoss()  # 创建一个交叉熵损失函数

    def repeat_state(self, state, csz, bsz, dim):  # 定义一个函数，用于重复状态
        return state.repeat(1, 2, 1).reshape(bsz, -1, csz, dim).transpose(1, 2).reshape(bsz, -1, dim)  # 返回重复后的状态

    def concat_states(self, hidden_states):  # 定义一个函数，用于连接状态
        bsz, _, dim = hidden_states[0].shape  # 获取hidden_states的第一个元素的形状
        states = []  # 创建一个空列表states
        for i in range(len(hidden_states)-1, 1, -1):  # 对于hidden_states的每一个元素，从后往前遍历
            state = hidden_states[i]  # 获取当前状态
            cur_csz = state.shape[1]  # 获取当前状态的第二个维度的大小
            for j in range(len(states)):  # 对于states的每一个元素
                states[j] = self.repeat_state(states[j], cur_csz, bsz, dim)[:, :hidden_states[i-1].shape[1]]  # 重复状态并取前一部分
            state = self.repeat_state(state, cur_csz, bsz, dim)[:, :hidden_states[i-1].shape[1]]  # 重复状态并取前一部分
            states.append(state)  # 将状态添加到states中
        states.append(hidden_states[1])  # 将hidden_states的第二个元素添加到states中
        return torch.concat(states[::-1], 2)  # 将states反转后进行连接，并返回


    def forward(self, data, pos, enc=True):
        '''
        data: bsz, context size, ancients + current node (4), level + octant + occ (3)
        '''
        padded = False
        if data.shape[1] % 2 == 1:
            padded = True
            pad = torch.zeros_like(data[:, :1])
            pad[:, :, :, 2] = 255
            data = torch.cat((data, pad), dim=1)
            pos_pad = torch.zeros_like(pos[:, :, :1])
            pos = torch.cat((pos, pos_pad), dim=2)

        bsz = data.shape[0]
        csz = data.shape[1]

        pre_occ = data[:, ::2, -1, -1]
        data = data.reshape(bsz, csz, -1)[:, :, :-1] # bsz, csz, 11. 11: 4*(level, oct, occ), except occ of current voxel

        feat = self.geo_feat_generator(data, pos)
        self_output = self.swin_self_transformer(feat, csz, output_hidden_states=True, output_hidden_states_before_downsampling=True)

        self_output = self.concat_states(self_output.hidden_states)
        feat_a = self.ancient_mlp(self_output)

        feat_a1 = feat_a[:, ::2]
        feat_a2 = feat_a[:, 1::2]
        prob1 = self.prob_pred_mlp1(feat_a1)

        pre_occ_embed = self.geo_feat_generator.embed_occ(pre_occ)
        pre_occ_feat = self.pre_occ_mlp(pre_occ_embed)
        pre_attn_feat = self.pre_attn_mlp(feat_a1)

        pre_feat = torch.concat((pre_occ_feat, pre_attn_feat), dim=2)
        cross_output = self.swin_cross_transformer(pre_feat, feat_a2.shape[1], query=feat_a2, output_hidden_states=True, output_hidden_states_before_downsampling=True)
        cross_output = self.concat_states(cross_output.hidden_states)
        feat_a2 = torch.concat((cross_output, feat_a2), 2)
        prob2 = self.prob_pred_mlp2(feat_a2)

        if padded:
            prob2 = prob2[:, :-1]

        if self.training:
            probs = torch.zeros((prob1.shape[0], prob1.shape[1] + prob2.shape[1], prob1.shape[2])).to(prob1.device)
            probs[:, ::2] = prob1
            probs[:, 1::2] = prob2
            return probs
        elif enc:
            return prob1, prob2

    def decode(self, data, pos, pre_occ=None):
        '''
        data: bsz, context size, ancients + current node (4), level + octant + occ (3)
        '''
        padded = False
        if data.shape[1] % 2 == 1:
            padded = True
            pad = torch.zeros_like(data[:, :1])
            pad[:, :, :, 2] = 255
            data = torch.cat((data, pad), dim=1)
            pos_pad = torch.zeros_like(pos[:, :, :1])
            pos = torch.cat((pos, pos_pad), dim=2)

        bsz = data.shape[0]
        csz = data.shape[1]

        if pre_occ is None:
            data = data.reshape(bsz, csz, -1)[:, :, :-1] # bsz, csz, 11. 11: 4*(level, oct, occ), except occ of current voxel

            feat = self.geo_feat_generator(data, pos)
            self_output = self.swin_self_transformer(feat, csz, output_hidden_states=True, output_hidden_states_before_downsampling=True)
            self_output = self.concat_states(self_output.hidden_states)
            feat_a = self.ancient_mlp(self_output)

            self.feat_a1 = feat_a[:, ::2]
            self.feat_a2 = feat_a[:, 1::2]
            prob1 = self.prob_pred_mlp1(self.feat_a1)
            return prob1

        pre_occ_embed = self.geo_feat_generator.embed_occ(pre_occ)
        pre_occ_feat = self.pre_occ_mlp(pre_occ_embed)
        pre_attn_feat = self.pre_attn_mlp(self.feat_a1)

        pre_feat = torch.concat((pre_occ_feat, pre_attn_feat), dim=2)
        cross_output = self.swin_cross_transformer(pre_feat, csz//2, query=self.feat_a2, output_hidden_states=True, output_hidden_states_before_downsampling=True)
        cross_output = self.concat_states(cross_output.hidden_states)
        feat_a2 = torch.concat((cross_output, self.feat_a2), 2)
        prob2 = self.prob_pred_mlp2(feat_a2)

        if padded:
            prob2 = prob2[:, :-1]

        return prob2

    def configure_optimizers(self):
        optim_cfg = self.cfg.train.optimizer
        sched_cfg = self.cfg.train.lr_scheduler
        if optim_cfg.name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr)
        else:
            raise NotImplementedError()
        if sched_cfg.name == "StepLR":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=sched_cfg.step_size, gamma=sched_cfg.gamma
            )
        else:
            raise NotImplementedError()

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, batch):
        data, pos, labels = batch
        if self.cfg.data.vari_data_len and torch.rand(1) < 0.3:
            sz = torch.randint(1, 8192, (1,))
            data = data[:, :sz]
            pos = pos[:, :, :sz]
            labels = labels[:, :sz]
        pred = self(data, pos)
        loss = self.criterion(
            pred.view(-1, self.cfg.model.token_num), labels.reshape(-1)
        ) / math.log(2)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def load_pretrain(self, path, strict=True):
        '''
        load EHEM pretrained model
        '''
        sd = torch.load(path)['state_dict']
        ref_sd = self.state_dict()
        keys = list(sd.keys())
        for k in keys:
            if k not in ref_sd or ref_sd[k].shape != sd[k].shape:
                sd.pop(k)
        return self.load_state_dict(sd, strict)

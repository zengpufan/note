## 一、WRIM论文解读
1. 文章解决了什么问题?
    
    本文解决了多模态人员重新识别（可见光和红外），即通过红外图像匹配可见光图像中的人。
    
2. 文章解决这个问题用了什么方法?
    
    文章提出了“Wide-Ranging Information Mining Network“（WRIM-Net），这个模型基于“Global Region Interaction (GRI)” ([pdf](zotero://open-pdf/library/items/Z47SD4D8?page=1))
    
    此外，文章还提出了一个新的对比学习方法，这种方法通过引入一个新的损失函数“Cross-Modality Key-Instance Contrastive (CMKIC) loss” ([pdf](zotero://open-pdf/library/items/Z47SD4D8?page=1))
    
3. 文章的创新点是什么?
    
    提出了“Multi-dimension Interactive Information Mining” ([pdf](zotero://open-pdf/library/items/Z47SD4D8?page=3)) 模块，该模块通过GRI模块进行增强
    
4. 模型的结构是什么（包括损失函数，优化器）?
    
    模型的损失函数分为三个部分：
    
    第一个部分是“CM KIC_P5_V I” ([pdf](zotero://open-pdf/library/items/Z47SD4D8?page=8)) ，这一部分是计算对比损失函数，将公式展开，除去一些常量，主要部分是可见光特征和正例红外光特征的向量积，减去可见光和反例红外光特征的向量积。
    
    总的对比损失函数是红外光对可见光的损失加上可见光对红外光的损失。
    
5. 用到的数据集是什么，结构是什么?
    
    本文用到了两个数据集，分别是SYSU-MM01和RegDB。
    
6. 前人的工作是什么?
    
    特征方法和图像方法。
    
    特征方法是将多模态图像的特征映射到同一个空间。
    
    图像方法是将不同模态的图像转换到同一个模态。
    
7. 作者做了哪些实验，评价指标是什么?
    
    这一任务的评价指标是 Rank1和mAP。
    
    作者在两个数据集上进行了实验，并且实验分为Indoor-search和all-search，分别表示室内图像和全部图像。此外实验还分为Single-shot和Multi-shot，分别表示每个人物训练一次和每个人物多次进行训练、
    
8. 模型的训练方法是什么?
    
    使用预训练的ResNet50作为backbone
    
9. 实验结果怎么样?


## 二、代码解析
1. 模型训练流程
```
train_net.py main(): 
    line 35: model = DefaultTrainer.build_model(cfg)
engine/defaults.py build_model():
    line 379: model = build_model(cfg)
code_train/fastreid/modeling/meta_arch/build.py :
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
```

2. 模型结构解析
![](./WRIM_Net/截屏2024-10-07%2017.32.50.png)
```py
# backbone of the MDI-Net
# 这段代码对应上图中的左侧部分
# 
class ResNet_MDI(ResNet):

    def __init__(self, last_stride, bn_norm, with_ibn, with_se, with_nl, block, layers, non_layers):
        super().__init__(last_stride, bn_norm, with_ibn, with_se, with_nl, block, layers, non_layers)
        num_heads = 8
        self.mdi1 = MDI_sep(num_heads=num_heads,channel=256,reduction=2,stride=4)
        self.mdi2 = MDI_sep(num_heads=num_heads,channel=512,reduction=2,stride=2)
        self.mdi3 = MDI(num_heads=num_heads,channel=1024,reduction=4,stride=1)
        self.mdi4 = MDI(num_heads=num_heads,channel=2048,reduction=4,stride=1) 

    def forward(self, x,mode=0):
        # x[128, 3, 384, 128]
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(len(self.layer1)):
            x = self.layer1[i](x)

        x = self.mdi1(x,mode)

        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
        layer2_out = x
        x = self.mdi2(x,mode)
        # x[128, 512, 48, 16]

        # x = self.msmc2(x)
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)

        x = self.mdi3(x)
        layer3_out = x 
        # x[128, 1024, 24, 8]
        # layer 4
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)

        x = self.mdi4(x)
        layer4_out = x
        # x[128, 2048, 24, 8]
        return {
            "layer2_out": layer2_out,
            "layer3_out": layer3_out,
            "layer4_out": layer4_out,
        }
```
MDINet(
  (backbone): ResNet_MDI(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (layer1): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
    (layer2): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (3): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
    (layer3): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (3): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (4): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (5): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
    (layer4): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
    (mdi1): MDI_sep(
      (si_v): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
      (si_i): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
      (sci_v): SCI(
        (scc): SpatialChannelCompress(
          (proj): Conv2d(256, 128, kernel_size=(4, 4), stride=(4, 4))
        )
        (mhsa): MHSA(
          (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (qkv): Linear(in_features=128, out_features=384, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=128, out_features=128, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (fc): Linear(in_features=128, out_features=256, bias=True)
        (sigmoid): Sigmoid()
        (upsample): Upsample(scale_factor=4.0, mode=nearest)
        (relu): ReLU(inplace=True)
        (si): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
      )
      (sci_i): SCI(
        (scc): SpatialChannelCompress(
          (proj): Conv2d(256, 128, kernel_size=(4, 4), stride=(4, 4))
        )
        (mhsa): MHSA(
          (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (qkv): Linear(in_features=128, out_features=384, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=128, out_features=128, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (fc): Linear(in_features=128, out_features=256, bias=True)
        (sigmoid): Sigmoid()
        (upsample): Upsample(scale_factor=4.0, mode=nearest)
        (relu): ReLU(inplace=True)
        (si): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
      )
    )
    (mdi2): MDI_sep(
      (si_v): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
      (si_i): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
      (sci_v): SCI(
        (scc): SpatialChannelCompress(
          (proj): Conv2d(512, 256, kernel_size=(2, 2), stride=(2, 2))
        )
        (mhsa): MHSA(
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (qkv): Linear(in_features=256, out_features=768, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=256, out_features=256, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (fc): Linear(in_features=256, out_features=512, bias=True)
        (sigmoid): Sigmoid()
        (upsample): Upsample(scale_factor=2.0, mode=nearest)
        (relu): ReLU(inplace=True)
        (si): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
      )
      (sci_i): SCI(
        (scc): SpatialChannelCompress(
          (proj): Conv2d(512, 256, kernel_size=(2, 2), stride=(2, 2))
        )
        (mhsa): MHSA(
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (qkv): Linear(in_features=256, out_features=768, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=256, out_features=256, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (fc): Linear(in_features=256, out_features=512, bias=True)
        (sigmoid): Sigmoid()
        (upsample): Upsample(scale_factor=2.0, mode=nearest)
        (relu): ReLU(inplace=True)
        (si): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
      )
    )
    (mdi3): MDI(
      (si): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
      (sci): SCI(
        (scc): SpatialChannelCompress(
          (proj): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        (mhsa): MHSA(
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (qkv): Linear(in_features=256, out_features=768, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=256, out_features=256, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (fc): Linear(in_features=256, out_features=1024, bias=True)
        (sigmoid): Sigmoid()
        (upsample): Upsample(scale_factor=1.0, mode=nearest)
        (relu): ReLU(inplace=True)
        (si): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
      )
    )
    (mdi4): MDI(
      (si): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
      (sci): SCI(
        (scc): SpatialChannelCompress(
          (proj): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
        )
        (mhsa): MHSA(
          (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (qkv): Linear(in_features=512, out_features=1536, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (fc): Linear(in_features=512, out_features=2048, bias=True)
        (sigmoid): Sigmoid()
        (upsample): Upsample(scale_factor=1.0, mode=nearest)
        (relu): ReLU(inplace=True)
        (si): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
      )
    )
  )
  (b1_head): ConEmbeddingHead(
    (pool_layer): GlobalAvgPool(output_size=1)
    (bottleneck): Sequential(
      (0): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (con_head): Sequential(
      (0): Linear(in_features=2048, out_features=2048, bias=True)
      (1): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Linear(in_features=2048, out_features=128, bias=True)
    )
    (cls_layer): Linear(num_classes=395, scale=1, margin=0.0)
  )
  (b11_head): ConEmbeddingHead(
    (pool_layer): GlobalAvgPool(output_size=1)
    (bottleneck): Sequential(
      (0): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (con_head): Sequential(
      (0): Linear(in_features=2048, out_features=2048, bias=True)
      (1): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Linear(in_features=2048, out_features=128, bias=True)
    )
    (cls_layer): Linear(num_classes=395, scale=1, margin=0.0)
  )
  (b12_head): ConEmbeddingHead(
    (pool_layer): GlobalAvgPool(output_size=1)
    (bottleneck): Sequential(
      (0): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (con_head): Sequential(
      (0): Linear(in_features=2048, out_features=2048, bias=True)
      (1): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Linear(in_features=2048, out_features=128, bias=True)
    )
    (cls_layer): Linear(num_classes=395, scale=1, margin=0.0)
  )
  (b2_head): ConEmbeddingHead(
    (pool_layer): GlobalAvgPool(output_size=1)
    (bottleneck): Sequential(
      (0): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (con_head): Sequential(
      (0): Linear(in_features=1024, out_features=1024, bias=True)
      (1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Linear(in_features=1024, out_features=128, bias=True)
    )
    (cls_layer): Linear(num_classes=395, scale=1, margin=0.0)
  )
)
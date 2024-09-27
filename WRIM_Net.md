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
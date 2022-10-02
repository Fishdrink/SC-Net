#  SC-Net:弱监督的表面缺陷检测
## 1.论文与代码  
* 本文：
  * 论文：暂未被录用
  * 代码：论文成功发表后公布（在基准论文代码上修改得来）
* 基准论文(2021年)：
  * 期刊：Computers in Industry(SCI影响因子11.245, Q1)
  * 作者：Božič J, Tabernik D, Skočaj D.
  * 论文：[Mixed supervision for surface-defect detection: From weakly to fully supervised learning](https://www.webofscience.com/wos/alldb/full-record/WOS:000648879500012)
  * 论文代码：[https://github.com/vicoslab/mixed-segdec-net-comind2021](https://github.com/vicoslab/mixed-segdec-net-comind2021)  
## 2.简述  
* 用途：工业表面缺陷检测（分类、分割）  
* 输入：
  * 数据集的样本图像
  * **图像级标签（二元分类标签，仅需标注是否存在缺陷,即True or False）**  
* 输出：
  * 分割图（分割出缺陷的轮廓与位置）  
  * 分类预测（样本图像存在缺陷的概率）  
* 实验数据集：KolektorSDD2(KSDD2), DAGM 1-10, KolektorSDD(KSDD), Severstal Steel（介绍与下载地址见文末）  

## 3.效果  
### (1) 表1 四个数据集的平均精确率AP  
| 论文 | 标签类型 | KSDD2 | DAGM | KSDD | Severstal Steel |  
| :---: | --- | :---: | :---: | :---: | :---: |  
| 本文 | 图像级标签（难度更高） | 96.0 | 100 | 99.4 | 96.4 |  
| 基准论文 | 图像级标签（难度更高） | 73.3 | 74.0 | 93.4 | 90.3 |  
| 基准论文 | 像素级/椭圆型/方框型标签 | 95.4 | 100 | 100 | 97.7 |  
* 注：
  * 平均精确率AP能更好地评估正负样本严重不平衡的缺陷检测类数据集
  * 实验记录还有其他评估指标，包括AUC, AC, F1及其阀值, FP, FN

### (2) 表2 模型大小与检测速度的比较（数据集KSDD2）  
| 论文 | 模型大小 | 训练速度 | 检测速度 | 训练所需迭代次数 | 训练所需时间 |  
| :---: | :---: | :---: | :---: | :---: | :---: |  
| 本文 | 2.4M | 9.8s/epoch | 125个样本/s | 91 | 20min |  
| 基准论文 | 59.7M | 53.5s/epoch | 33个样本/s | 31 | 30min |  

### (3) 分割图（KSDD）
<img src="https://user-images.githubusercontent.com/65808993/192088013-20774c83-fed0-41b8-8d88-7ff72629c346.png" width="355px" height="600px">

## 4.数据集  
### (1) 简介与下载地址  
* KolektorSDD2。
  * 该数据集由有缺陷的电气换向器的彩色图像构成，由Kolektor Group doo提供并部分注释，由视觉检查系统捕获，在受控环境中捕获的图像大小相似，大约230 像素宽和 630 像素高。数据集分为训练和测试子集，训练中有2085个负样本和246个正样本，测试子集中有894个负样本和110个正样本。缺陷用细粒度的分割掩码标注，形状、大小和颜色各不相同，从小划痕和小斑点到大表面缺陷不等。
  * [下载地址](https://www.vicos.si/resources/kolektorsdd2/)
* DAGM。
  * 该数据集是众所周知的表面缺陷检测基准数据集。它包含十种不同的计算机生成表面和各种缺陷（如划痕或斑点）的灰度图像。每个表面都被视为一个二元分类问题。最初公开了六个类别，后来又公开了四个类别；因此，一些相关方法只报告前六个类的结果，而其他方法则报告所有十个类的结果。
  * [下载地址](https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-optical-inspection)
* KolektorSDD。
  * 该数据集包含真实世界生产项目的灰度图像；其中许多包含可见的表面裂缝。由于样本量小，图像被分成三部分，如Tabernik、Šela 等人。（2019），而最终结果报告为三倍交叉验证的平均值。
  * [下载地址](https://www.vicos.si/resources/kolektorsdd/)
* Severstal Steel。
  * 该缺陷数据集明显大于其他三个数据集，包含 4 类 12,568 张灰度图像，具有各种缺陷。我们在评估中仅使用数据集的一个子集。特别是，我们使用所有负片图像，但只考虑图像中存在最常见缺陷类别的正片图像（第 3 类）。缺陷在尺寸、形状和外观上非常多样化，从划痕和凹痕到多余的材料。尽管数据集相当大且多样化，但一些缺陷非常模糊，可能无法正确注释。
  * [下载地址](https://www.kaggle.com/c/severstal-steel-defect-detection/data)
### (2) 表3 数据集的结构 
| 数据集 | 类别 | KSDD2 | DAGM | KSDD | Severstal Steel |  
| :---: | :---: | :---: | :---: | :---: | :---: |  
| 训练集 | 正样本 | 246 | 1046 | 34 | 300 |  
|        | 负样本 | 2086 | 7004 | 230 | 4143 |  
| 验证集 | 正样本 | 110 | 1054 | 110 | 559 |  
|        | 负样本 | 894 | 6996 | 894 | 559 |  
| 测试集 | 正样本 | 110 | 1054 | 110 | 1200 |  
|        | 负样本 | 894 | 6996 | 894 | 1200 |  
* 注：
  * 含缺陷的图像为正样本，无缺陷图像为负样本
  * 前3个数据集并没有真正的验证集，使用测试集代替

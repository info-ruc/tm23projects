## 项目运行：
### 1.下载预训练模型文件
首先将ERNIE模型下载放到yQuery\pretrain_models\ERNIEM目录下。链接：https://pan.baidu.com/s/1fdryG3qetgv99-hAz8yCwQ?pwd=duat 提取码：duat

### 2. 程序运行命令：
```
cd code
#运行增强文件脚本
python data_augment.py
#运行训练脚本
python run_ernie.py
#运行测试脚本
python test.py
```

### 3.模型运行结果
生成后的模型文件：
链接：https://pan.baidu.com/s/1p4dWcNb9_d2tnCRNt17how?pwd=3rbt 提取码：3rbt

运行日志可查看训练时的得分：logging/ernie

生成后的预测结果：prediction_result/ernie_pred.json

## 一、摘要
医学搜索词相关性判断项目通过处理医学文本数据，识别医学领域Query之间的转义及其程度，以提高搜索效率与准确性、改善用户体验等。该项目主要包括：数据收集与准备、query特征提取、模型的选择（ERNIE模型）与训练、逻辑推理模型应用、测试与部署等。该项目综合运用自然语言处理、机器学习和医学领域知识，提供更准确、更全面、更相关的医学信息支持，满足用户在医学领域信息检索的需求，有助于提高用于检索的准确性。

## 二、前言
医学搜索query指的是用户在医学领域进行信息检索时输入的搜索词或查询。这些查询通常包含与健康、疾病、症状、治疗、药物等相关的术语或关键词，用于寻找与健康、医疗或特定疾病相关的信息。如果Query-A和Query-B表示相似的医学概念、症状、疾病或其他相关内容，尽管表达方式有所不同，但它们可能具有等效或相似的含义。

判断用户输入的搜索词之间的相关性及程度意义重大。例如，当用户输入"小孩子打呼噜什么原因"时，如果医学搜索引擎能够判断出"小孩子打呼噜是什么原因引起的"与之相关性很高，那么就可以将相关的搜索结果展示给用户，从而提高用户的检索效率和体验。相反，如果医学搜索引擎将"小孩子打呼噜"和"孩子打呼噜"判断为相关性很高，那么就可能导致用户得到不准确或不完整的信息，从而降低用户的信任度和满意度。

我们开展的医学搜索词相关性判断项目旨在通过结合数据科学、自然语言处理、机器学习等多个领域先进的技术手段与知识，识别医学领域Query之间的转义及其程度。项目成果可为医学领域的信息检索和诊断提供支撑，提升医学信息检索的质量、用户体验，并对医疗决策和学术研究等方面产生积极影响。

## 三、数据预处理
### 1.存储新生成的问题对和标签
首先定义了三个空列表questions1，questions2和labels，用于存储新生成的问题对和标签。然后将训练数据集train_examples转换为DataFrame格式，并获取其中唯一的query1值列表。

### 2.存储相似度
对每个唯一的query1值进行遍历，然后根据相似度标签（label）将相应的query2值分别存储到query_same_set、query_sim_set和query_diff_set列表中。

### 3.生成新的问题对和标签
根据不同的情况，生成新的问题对和标签。具体来说，如果query_same_set中有问题，我们会根据不同的相似度标签生成新的问题对，并将其添加到questions1、questions2和labels列表中。
接着将新生成的问题对和标签转换为DataFrame格式，并根据标签值将数据分为三个子集：df_postive（类别内相似）、df_negative（类别间不相似）和df_similar（相似）。然后对三个子集进行采样，以控制各类别数据的比例。

### 4.返回增强数据集的列表形式
将采样后的数据集合并成一个新的数据集new_df，如果指定了文件名file_name，则将增强后的数据集保存为CSV文件，并返回增强数据集的列表形式。

## 四、训练
### 1.加载预训练模型ERNIE
加载预训练模型ERNIE，共三个文件：pytorch模型权重、配置文件、词表。ERNIE默认配置：隐藏层激活函数  relu；隐藏层大小 768；隐藏层层数 12；注意力头数 12；最大位置编码长度 513；隐藏层和注意力输出的随机失活概率都为0.1；层归一化的epsilon为1e-5等

### 2.配置训练测试数据路径
四份数据，train、dev、test、训练增强augment(基于逻辑三段论进行扩充)。数据集自变量为查询文本对 (query1 和 query2)，目标变量为标签值（0：语义不同；1：语义相似；2：语义相同。）；

### 3.配置ERNIE参数
隐藏层的 dropout 概率 /隐藏层大小与默认值一致，不采用early_stop策略；epochs迭代次数10；训练批次大小128；每句话处理长度64；学习率1e-5；分类层学习率1e-3；权重衰减因子0.01等

### 4.配置参数到模型
通过BertTokenizer.from_pretrained从ernie模型加载中文词表和分词器，赋给变量 tokenizer。
通过BertConfig.from_pretrained加载上述ERNIE配置的参数，并设置为三分类任务，赋给变量 model_config
通过BertModel.from_pretrained加载ERINE权重模型和上步的配置参数model_config，赋给变量bert
配置神经网络分类器

### 5.前向传播函数forward
输入文本向量input_ids，注意力掩码attention_mask，句子标识token_type_ids，目标变量labels，输出out和loss。

将所有入参输入模型bert后得到output，通过output[1]得到整个输入序列的一个池化表示，再经随机失活和分类器预测得到目标值，通过交叉熵损失函数计算损失，循环5次，取损失的平均值

### 6.训练
优化器设置：配置AdamW优化器，基于模型的不同部分指定不同的学习率。

学习率调整：通过线性预热和衰减调度来调整学习率，学习率从一个预热阶段开始，逐渐增加，然后在训练结束时线性减小。

循环训练：对整个训练数据循环10次（10 Epoch），每次训练分批次（batch）处理。通过模型前向传播计算输出和损失。

梯度计算和更新： 计算损失后，进行反向传播，计算梯度，并通过优化器更新模型参数。同时，使用学习率调度器（scheduler）进行学习率的更新。

性能评估： 每隔一定的 batch 数量，评估模型在训练集上的准确率（train_acc），并在验证集上评估模型性能（准确率 和平均损失值）。通过metrics.accuracy_score评估准确率。

模型保存： 如果当前模型在验证集上的准确率超过之前的最佳准确率，并且训练集上的准确率超过阈值（0.85），则保存当前模型为最佳模型。同时，记录训练和验证的准确率、损失以及经过的时间等信息。

## 五、推理
将测试数据预处理后，输入上述训练的模型，得到预测结果，并写入json文件。

训练成绩：0.9148 测试成绩：0.8290。

## 六、结论
通过本项目的研究，项目组成功利用自然语言处理技术实现了医学搜索词之间的相关性分析，提高检索的准确性。通过数据预处理、模型训练、参数配置、训练调优、推理等步骤，构建了医学搜索词之间的相关性分析模型，该模型准确率指标达到了较高的水平。

通过本项目的研究，小组成员实践了自然语言处理、transfomer、ERNIE模型等技术，在医学搜索词之间的相关性分析的应用，小组成员掌握了文本挖掘课程所学到的自然语言技术实践经验。

## 环境配置：
* GPU 阿里云V100*16G
* ubuntu 20.04.1
* cuda == 11.3
* python == 3.8.13 
* pytorch == 1.10.1 
* transformers==4.21.1   
* numpy==1.22.4

## 项目成员与分工：
2023年秋大数据科学与技术

苏莹：队长，负责组队分工和后勤支持，远程代码调试，参与项目报告编写

宋银州：负责编码、训练、调试、优化，计算和资源环境准备、部署，主导项目报告编写

张志祥：负责开题汇报，远程代码调试，参与项目报告编写

田炳河：负责开题汇报文档编写，远程代码调试，参与项目报告编写

## 参考项目：
https://github.com/renqi1/KUAKE_Query_Relevance
https://github.com/thunderboom/text_similarity

## 测试结果：
INFO: ***** Running training *****
INFO: Train Num examples = 37695
INFO: Dev Num examples = 1600
INFO: Num Epochs = 10
INFO: Instantaneous batch size GPU/CPU = 128
INFO: Total optimization steps = 2950
INFO: Train device:cuda
INFO: Iter: 20/ 295, epoch: 1/ 10, Train Loss: 0.514682, Train Acc: 25.04%, Val Loss: 0.541336, Val Acc: 17.56%, Time: 5.236645936965942
INFO: Iter: 40/ 295, epoch: 1/ 10, Train Loss: 0.507253, Train Acc: 28.16%, Val Loss: 0.509831, Val Acc: 20.19%, Time: 10.462707996368408
INFO: Iter: 60/ 295, epoch: 1/ 10, Train Loss: 0.472357, Train Acc: 30.59%, Val Loss: 0.467983, Val Acc: 30.06%, Time: 15.690887928009033
INFO: Iter: 80/ 295, epoch: 1/ 10, Train Loss: 0.437212, Train Acc: 36.37%, Val Loss: 0.425243, Val Acc: 51.75%, Time: 20.924217462539673
INFO: Iter: 100/ 295, epoch: 1/ 10, Train Loss: 0.407420, Train Acc: 42.34%, Val Loss: 0.389074, Val Acc: 64.44%, Time: 26.196054220199585
INFO: Iter: 120/ 295, epoch: 1/ 10, Train Loss: 0.387516, Train Acc: 46.88%, Val Loss: 0.366934, Val Acc: 68.44%, Time: 31.46281909942627
INFO: Iter: 140/ 295, epoch: 1/ 10, Train Loss: 0.383866, Train Acc: 49.57%, Val Loss: 0.347731, Val Acc: 72.69%, Time: 36.73541331291199
INFO: Iter: 160/ 295, epoch: 1/ 10, Train Loss: 0.368946, Train Acc: 54.26%, Val Loss: 0.329360, Val Acc: 74.44%, Time: 41.981722831726074
INFO: Iter: 180/ 295, epoch: 1/ 10, Train Loss: 0.341975, Train Acc: 58.95%, Val Loss: 0.300962, Val Acc: 76.38%, Time: 47.28019857406616
INFO: Iter: 200/ 295, epoch: 1/ 10, Train Loss: 0.363696, Train Acc: 65.51%, Val Loss: 0.278162, Val Acc: 75.94%, Time: 52.58547830581665
INFO: Iter: 220/ 295, epoch: 1/ 10, Train Loss: 0.333632, Train Acc: 67.27%, Val Loss: 0.263965, Val Acc: 75.88%, Time: 57.84825682640076
INFO: Iter: 240/ 295, epoch: 1/ 10, Train Loss: 0.276114, Train Acc: 71.13%, Val Loss: 0.251867, Val Acc: 77.94%, Time: 63.12018799781799
INFO: Iter: 260/ 295, epoch: 1/ 10, Train Loss: 0.259242, Train Acc: 71.80%, Val Loss: 0.231304, Val Acc: 80.31%, Time: 68.36094737052917
INFO: Iter: 280/ 295, epoch: 1/ 10, Train Loss: 0.250225, Train Acc: 71.99%, Val Loss: 0.226663, Val Acc: 80.19%, Time: 73.67360234260559
INFO: Iter: 0/ 295, epoch: 2/ 10, Train Loss: 0.221433, Train Acc: 74.67%, Val Loss: 0.221351, Val Acc: 80.25%, Time: 78.83759117126465
INFO: Iter: 20/ 295, epoch: 2/ 10, Train Loss: 0.235742, Train Acc: 74.69%, Val Loss: 0.211583, Val Acc: 80.00%, Time: 84.1343743801117
INFO: Iter: 40/ 295, epoch: 2/ 10, Train Loss: 0.215987, Train Acc: 78.24%, Val Loss: 0.195160, Val Acc: 82.44%, Time: 89.44626355171204
INFO: Iter: 60/ 295, epoch: 2/ 10, Train Loss: 0.239908, Train Acc: 77.50%, Val Loss: 0.207567, Val Acc: 80.88%, Time: 94.74806094169617
INFO: Iter: 80/ 295, epoch: 2/ 10, Train Loss: 0.226242, Train Acc: 78.24%, Val Loss: 0.205121, Val Acc: 81.00%, Time: 100.04860997200012
INFO: Iter: 100/ 295, epoch: 2/ 10, Train Loss: 0.216857, Train Acc: 77.58%, Val Loss: 0.192850, Val Acc: 82.69%, Time: 105.34504842758179
INFO: Iter: 120/ 295, epoch: 2/ 10, Train Loss: 0.221465, Train Acc: 79.49%, Val Loss: 0.186433, Val Acc: 83.62%, Time: 110.64391350746155
INFO: Iter: 140/ 295, epoch: 2/ 10, Train Loss: 0.175457, Train Acc: 79.73%, Val Loss: 0.208186, Val Acc: 80.19%, Time: 115.93146061897278
INFO: Iter: 160/ 295, epoch: 2/ 10, Train Loss: 0.151042, Train Acc: 81.29%, Val Loss: 0.181847, Val Acc: 83.44%, Time: 121.24468469619751
INFO: Iter: 180/ 295, epoch: 2/ 10, Train Loss: 0.228120, Train Acc: 80.70%, Val Loss: 0.173897, Val Acc: 84.06%, Time: 126.52738308906555
INFO: Iter: 200/ 295, epoch: 2/ 10, Train Loss: 0.195341, Train Acc: 80.70%, Val Loss: 0.172664, Val Acc: 83.94%, Time: 131.8205063343048
INFO: Iter: 220/ 295, epoch: 2/ 10, Train Loss: 0.279501, Train Acc: 82.27%, Val Loss: 0.186278, Val Acc: 82.19%, Time: 137.09972643852234
INFO: Iter: 240/ 295, epoch: 2/ 10, Train Loss: 0.177972, Train Acc: 81.84%, Val Loss: 0.171421, Val Acc: 84.12%, Time: 142.51874113082886
INFO: Iter: 260/ 295, epoch: 2/ 10, Train Loss: 0.175121, Train Acc: 82.03%, Val Loss: 0.182593, Val Acc: 82.75%, Time: 147.83114767074585
INFO: Iter: 280/ 295, epoch: 2/ 10, Train Loss: 0.192487, Train Acc: 82.46%, Val Loss: 0.178767, Val Acc: 83.81%, Time: 153.14609026908875
INFO: Iter: 0/ 295, epoch: 3/ 10, Train Loss: 0.140361, Train Acc: 83.21%, Val Loss: 0.177962, Val Acc: 83.19%, Time: 158.34488439559937
INFO: Iter: 20/ 295, epoch: 3/ 10, Train Loss: 0.207186, Train Acc: 84.41%, Val Loss: 0.166074, Val Acc: 84.56%, Time: 163.631653547287
INFO: Iter: 40/ 295, epoch: 3/ 10, Train Loss: 0.187121, Train Acc: 82.77%, Val Loss: 0.179097, Val Acc: 82.56%, Time: 168.92683100700378
INFO: Iter: 60/ 295, epoch: 3/ 10, Train Loss: 0.173847, Train Acc: 84.22%, Val Loss: 0.169394, Val Acc: 84.12%, Time: 174.23075938224792
INFO: Iter: 80/ 295, epoch: 3/ 10, Train Loss: 0.203490, Train Acc: 84.73%, Val Loss: 0.175252, Val Acc: 83.88%, Time: 179.49092364311218
INFO: model saved, path: ../my_model/best_ernie.pkl
INFO: Iter: 100/ 295, epoch: 3/ 10, Train Loss: 0.206372, Train Acc: 85.62%, Val Loss: 0.175551, Val Acc: 83.44%, Time: 185.17589282989502 *
INFO: Iter: 120/ 295, epoch: 3/ 10, Train Loss: 0.241940, Train Acc: 83.75%, Val Loss: 0.188131, Val Acc: 82.38%, Time: 190.48119568824768
INFO: Iter: 140/ 295, epoch: 3/ 10, Train Loss: 0.145568, Train Acc: 86.80%, Val Loss: 0.184662, Val Acc: 83.00%, Time: 195.7308909893036
INFO: Iter: 160/ 295, epoch: 3/ 10, Train Loss: 0.198614, Train Acc: 86.37%, Val Loss: 0.179001, Val Acc: 83.44%, Time: 201.03361010551453
INFO: Iter: 180/ 295, epoch: 3/ 10, Train Loss: 0.184965, Train Acc: 85.86%, Val Loss: 0.175678, Val Acc: 83.31%, Time: 206.34392142295837
INFO: Iter: 200/ 295, epoch: 3/ 10, Train Loss: 0.115026, Train Acc: 86.68%, Val Loss: 0.184091, Val Acc: 82.31%, Time: 211.63353848457336
INFO: Iter: 220/ 295, epoch: 3/ 10, Train Loss: 0.170294, Train Acc: 85.08%, Val Loss: 0.178503, Val Acc: 82.81%, Time: 216.95318841934204
INFO: Iter: 240/ 295, epoch: 3/ 10, Train Loss: 0.153009, Train Acc: 85.59%, Val Loss: 0.175301, Val Acc: 82.81%, Time: 222.2351324558258
INFO: Iter: 260/ 295, epoch: 3/ 10, Train Loss: 0.114629, Train Acc: 85.98%, Val Loss: 0.189623, Val Acc: 81.56%, Time: 227.5511486530304
INFO: Iter: 280/ 295, epoch: 3/ 10, Train Loss: 0.139580, Train Acc: 86.29%, Val Loss: 0.179228, Val Acc: 83.00%, Time: 232.86815118789673
INFO: Iter: 0/ 295, epoch: 4/ 10, Train Loss: 0.148346, Train Acc: 86.57%, Val Loss: 0.183785, Val Acc: 82.12%, Time: 238.06853461265564
INFO: Iter: 20/ 295, epoch: 4/ 10, Train Loss: 0.149697, Train Acc: 88.55%, Val Loss: 0.184434, Val Acc: 82.56%, Time: 243.3748013973236
INFO: Iter: 40/ 295, epoch: 4/ 10, Train Loss: 0.160224, Train Acc: 89.10%, Val Loss: 0.202026, Val Acc: 80.81%, Time: 248.69325518608093
INFO: Iter: 60/ 295, epoch: 4/ 10, Train Loss: 0.117803, Train Acc: 86.99%, Val Loss: 0.188216, Val Acc: 82.38%, Time: 254.00271487236023
INFO: Iter: 80/ 295, epoch: 4/ 10, Train Loss: 0.123081, Train Acc: 87.54%, Val Loss: 0.187293, Val Acc: 82.19%, Time: 259.3363230228424
INFO: Iter: 100/ 295, epoch: 4/ 10, Train Loss: 0.152797, Train Acc: 87.19%, Val Loss: 0.181465, Val Acc: 83.00%, Time: 264.6344451904297
INFO: Iter: 120/ 295, epoch: 4/ 10, Train Loss: 0.148591, Train Acc: 87.70%, Val Loss: 0.201006, Val Acc: 81.44%, Time: 269.9373097419739
INFO: Iter: 140/ 295, epoch: 4/ 10, Train Loss: 0.165710, Train Acc: 88.01%, Val Loss: 0.204817, Val Acc: 81.19%, Time: 275.2206726074219
INFO: Iter: 160/ 295, epoch: 4/ 10, Train Loss: 0.121012, Train Acc: 86.99%, Val Loss: 0.192558, Val Acc: 81.75%, Time: 280.5320179462433
INFO: Iter: 180/ 295, epoch: 4/ 10, Train Loss: 0.138342, Train Acc: 88.01%, Val Loss: 0.191547, Val Acc: 81.94%, Time: 285.8404664993286
INFO: Iter: 200/ 295, epoch: 4/ 10, Train Loss: 0.169909, Train Acc: 87.27%, Val Loss: 0.184831, Val Acc: 83.25%, Time: 291.14758706092834
INFO: Iter: 220/ 295, epoch: 4/ 10, Train Loss: 0.117483, Train Acc: 88.79%, Val Loss: 0.180015, Val Acc: 83.31%, Time: 296.46068048477173
INFO: Iter: 240/ 295, epoch: 4/ 10, Train Loss: 0.097993, Train Acc: 89.77%, Val Loss: 0.188474, Val Acc: 82.44%, Time: 301.7680125236511
INFO: Iter: 260/ 295, epoch: 4/ 10, Train Loss: 0.114280, Train Acc: 86.99%, Val Loss: 0.183651, Val Acc: 82.25%, Time: 307.0882611274719
INFO: Iter: 280/ 295, epoch: 4/ 10, Train Loss: 0.121835, Train Acc: 88.26%, Val Loss: 0.183857, Val Acc: 82.19%, Time: 312.3136565685272
INFO: Iter: 0/ 295, epoch: 5/ 10, Train Loss: 0.067489, Train Acc: 90.00%, Val Loss: 0.187464, Val Acc: 83.00%, Time: 317.63819336891174
INFO: Iter: 20/ 295, epoch: 5/ 10, Train Loss: 0.073693, Train Acc: 89.18%, Val Loss: 0.187543, Val Acc: 82.75%, Time: 322.9565007686615
INFO: Iter: 40/ 295, epoch: 5/ 10, Train Loss: 0.161657, Train Acc: 89.65%, Val Loss: 0.189227, Val Acc: 82.50%, Time: 328.2611904144287
INFO: Iter: 60/ 295, epoch: 5/ 10, Train Loss: 0.130674, Train Acc: 89.02%, Val Loss: 0.188615, Val Acc: 82.50%, Time: 333.69745802879333
INFO: Iter: 80/ 295, epoch: 5/ 10, Train Loss: 0.081355, Train Acc: 91.48%, Val Loss: 0.194852, Val Acc: 82.31%, Time: 339.0295829772949
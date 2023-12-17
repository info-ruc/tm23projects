#!/usr/bin/env python
# coding: utf-8

# In[1]:
import collections
import math
import random
import sys
import time
import os
import numpy as np
import tensorflow as tf 

sys.path.append("../..")


# In[2]:
assert 'ptb.train.txt' in os.listdir("E:\科研类研究\数据集\PTE文本数据集\simple-examples\data")


# In[4]:
with open('E:\科研类研究\数据集\PTE文本数据集\simple-examples\data\ptb.train.txt', 'r') as f:
    lines = f.readlines()
    # st是sentence的缩写
    raw_dataset = [st.split() for st in lines]

'# sentences: %d' % len(raw_dataset)


# In[5]:
with open('E:\科研类研究\数据集\PTE文本数据集\simple-examples\data\ptb.train.txt', 'r') as f:
    lines = f.readlines()
    # st是sentence的缩写
    raw_dataset = [st.split() for st in lines]

'# sentences: %d' % len(raw_dataset)


# In[7]:
for st in raw_dataset[:3]:
    print('# tokens:', len(st), st[:5])


# In[8]:
counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))


# In[9]:
idx_to_token = [tk for tk, _ in counter.items()]
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx]
           for st in raw_dataset]
num_tokens = sum([len(st) for st in dataset])
'# tokens: %d' % num_tokens


# In[10]:
def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(
        1e-4 / counter[idx_to_token[idx]] * num_tokens)

subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
'# tokens: %d' % sum([len(st) for st in subsampled_dataset])


# In[11]:
def compare_counts(token):
    return '# %s: before=%d, after=%d' % (token, sum(
        [st.count(token_to_idx[token]) for st in dataset]), sum(
        [st.count(token_to_idx[token]) for st in subsampled_dataset]))

compare_counts('the')


# In[12]:
compare_counts('join')


# In[13]:
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts


# In[14]:
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)


# In[15]:
ll_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)


# In[16]:
all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)


# In[17]:
def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                # 为了高效计算，可以将k设得稍大一点
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i + 1
            # 噪声词不能是背景词
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

sampling_weights = [counter[w]**0.75 for w in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)


# In[18]:
def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        center=center.numpy().tolist()
        context=context.numpy().tolist()
        negative=negative.numpy().tolist()
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return tf.data.Dataset.from_tensor_slices((tf.reshape(tf.convert_to_tensor(centers),shape=(-1, 1)), tf.convert_to_tensor(contexts_negatives),
            tf.convert_to_tensor(masks), tf.convert_to_tensor(labels)))


# In[19]:
def generator():
    for cent, cont, neg in zip(all_centers,all_contexts,all_negatives):
        yield (cent, cont, neg)


# In[20]:
batch_size = 512
dataset=tf.data.Dataset.from_generator(generator=generator,output_types=(tf.int32,tf.int32, tf.int32))
dataset = dataset.apply(batchify).shuffle(len(all_centers)).batch(batch_size)


# In[21]:
for batch in dataset:
    for name, data in zip(['centers', 'contexts_negatives', 'masks',
                           'labels'], batch):
        print(name, 'shape:', data.shape)
    break


# In[22]:
embed = tf.keras.layers.Embedding(input_dim=20, output_dim=4)
embed.build(input_shape=(1,20))
embed.get_weights()


# In[23]:
x = tf.convert_to_tensor([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
embed(x)


# In[24]:
X = tf.ones((2, 1, 4))
Y = tf.ones((2, 4, 6))
tf.matmul(X, Y).shape


# In[25]:
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = tf.matmul(v, tf.transpose(u,perm=[0,2,1]))
    return pred


# In[26]:
class SigmoidBinaryCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self): # none mean sum
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
    def __call__(self, inputs, targets, mask=None):
        #tensorflow中使用tf.nn.weighted_cross_entropy_with_logits设置mask并没有起到作用
        #直接与mask按元素相乘回实现当mask为0时不计损失的效果
        inputs=tf.cast(inputs,dtype=tf.float32)
        targets=tf.cast(targets,dtype=tf.float32)
        mask=tf.cast(mask,dtype=tf.float32)
        res=tf.nn.sigmoid_cross_entropy_with_logits(inputs, targets)*mask
        return tf.reduce_mean(res,axis=1)

loss = SigmoidBinaryCrossEntropyLoss()


# In[27]:
pred = tf.convert_to_tensor([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]],dtype=tf.float32)
# 标签变量label中的1和0分别代表背景词和噪声词
label = tf.convert_to_tensor([[1, 0, 0, 0], [1, 1, 0, 0]],dtype=tf.float32)
mask = tf.convert_to_tensor([[1, 1, 1, 1], [1, 1, 1, 0]],dtype=tf.float32)  # 掩码变量
loss(label, pred, mask) * mask.shape[1] / tf.reduce_sum(mask,axis=1)


# In[28]:
def sigmd(x):
    return - math.log(1 / (1 + math.exp(-x)))
print('%.4f' % ((sigmd(1.5) + sigmd(-0.3) + sigmd(1) + sigmd(-2)) / 4)) # 注意1-sigmoid(x) = sigmoid(-x)
print('%.4f' % ((sigmd(1.1) + sigmd(-0.6) + sigmd(-2.2)) / 3))


# In[29]:
embed_size = 100
net = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(idx_to_token), output_dim=embed_size),
    tf.keras.layers.Embedding(input_dim=len(idx_to_token), output_dim=embed_size)
])
net.get_layer(index=0)

# In[30]:
def train(net, lr, num_epochs):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in dataset:
            center, context_negative, mask, label = [d for d in batch]
            mask=tf.cast(mask,dtype=tf.float32)
            with tf.GradientTape(persistent=True) as tape:
                pred = skip_gram(center, context_negative, net.get_layer(index=0), net.get_layer(index=1))
                # 使用掩码变量mask来避免填充项对损失函数计算的影响
                l = (loss(label, tf.reshape(pred,label.shape), mask) *
                     mask.shape[1] / tf.reduce_sum(mask,axis=1))
                l=tf.reduce_mean(l)# 一个batch的平均loss
                
            grads = tape.gradient(l, net.variables)
            optimizer.apply_gradients(zip(grads, net.variables))
            l_sum += np.array(l).item()
            n += 1
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))



# In[31]:
def get_similar_tokens(query_token, k, embed):
    W = embed.get_weights()
    W = tf.convert_to_tensor(W[0])
    x = W[token_to_idx[query_token]]
    x = tf.reshape(x,shape=[-1,1])
    # 添加的1e-9是为了数值稳定性
    cos = tf.reshape(tf.matmul(W, x),shape=[-1])/ tf.sqrt(tf.reduce_sum(W * W, axis=1) * tf.reduce_sum(x * x) + 1e-9)
    _, topk = tf.math.top_k(cos, k=k+1)
    topk=topk.numpy().tolist()
    for i in topk[1:]:  # 除去输入词
        print('cosine sim=%.3f: %s' % (cos[i], (idx_to_token[i])))






get_similar_tokens('computer', 3, net.get_layer(index=0))






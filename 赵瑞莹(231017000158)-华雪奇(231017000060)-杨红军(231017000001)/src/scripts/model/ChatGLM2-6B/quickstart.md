---
<!-- 该部分为参数配置部分 -->

---
<!-- 公共内容部分  -->

## 模型加载和推理
更多关于模型加载和推理的问题参考[模型的推理Pipeline](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%8E%A8%E7%90%86Pipeline)。

```python
from modelscope.utils.constant import Tasks
from modelscope import Model
from modelscope.pipelines import pipeline
model = Model.from_pretrained('ZhipuAI/chatglm2-6b', device_map='auto', revision='v1.0.6')
pipe = pipeline(task=Tasks.chat, model=model)
inputs = {'text':'你好', 'history': []}
result = pipe(inputs)
inputs = {'text':'介绍下江南大学', 'history': result['history']}
result = pipe(inputs)
print(result)
```

更多使用说明请参阅[ModelScope文档中心](http://www.modelscope.cn/#/docs)。

---
<!-- 在线使用独有内容部分 -->

---
<!-- 本地使用独有内容部分 -->
## 下载并安装ModelScope library
更多关于下载安装ModelScope library的问题参考[环境安装](https://modelscope.cn/docs/%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85)。

```python
pip install "modelscope[audio,cv,nlp,multi-modal,science]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

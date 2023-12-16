from sentence_transformers import SentenceTransformer, util
# 选用sbert的 预训练模型
#model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
# 将模型保存到本地
#model.save("E:/localstorage/model/paraphrase-MiniLM-L3-v2")
# 使用保存到本地的模型
model = SentenceTransformer('E:/localstorage/model/paraphrase-MiniLM-L3-v2')
query_embedding = model.encode('How big is London')
passage_embedding = model.encode(['London has 9,787,426 inhabitants at the 2011 census',
                                  'London is known for its finacial district'])

print("Similarity:", util.dot_score(query_embedding, passage_embedding))

from sentence_transformers import SentenceTransformer, util
#model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('E:/localstorage/model/paraphrase-MiniLM-L3-v2')

from sentence_transformers import SentenceTransformer, util
import torch

embedder = SentenceTransformer('E:/localstorage/model/paraphrase-MiniLM-L3-v2')

# Corpus with example sentences
# corpus = ["中国是怎么实现四个现代化的",
#           "中国人喜欢吃米饭",
#           "中国的国旗是什么样子的"
#           ]
from datasets import load_from_disk
#加载数据集
#使用在Huggingface下载的知乎kol数据集，该数据集包含了问题、回答、该问题对应的知乎的问题ID，问题的URL等
dataset = load_from_disk("E:/localstorage/dataset/wangrui6_Zhihu-KOL")["train"]
# corpus = dataset.to_list()
corpus = [{'INSTRUCTION': '二代基因测序准确吗?', 'RESPONSE': '基因测序无论如何都不可能比表型直接检测更可靠。从基因到表型过程中，首先由表观遗传学标记，有转录、转录后修饰、翻译、翻译后修饰诸多环节，最终才能得到一个表型。用通俗的话来说，一个基因到其所表达的产物的路径是极其复杂的，就算你检测到一个基因，你也不能确保这个基因能产生出你所观察到的致病或者健康的产物。因此基因诊断难以像直接检验这些产物那么有说服力，而穿刺活检会能更直接地说明病情和病灶。 一个疾病是一个极其复杂的性状，往往由成千上万个基因调控，并不是所有基因的功能都是已知的，一般基因测序只测或者只能识别出已经知道并且研究较为透彻的位点，这些位点往往更有可能有人正在或者已经研发出靶向药物。检测不出来说明致病机理不明确、缺乏靶向药物，并非说明没有病。二代测序不如第一代测序准确，但是胜在可以读取大量位点，性价比较高。但已知的致病基因就那么多，靶向药物不可能针对未知基因开发，用哪一代测序都有可能有检测不出来的情况。 提醒：别把知乎当做征求医学建议的地方，有问题应该和医生商量。', 'SOURCE': 'Zhihu', 'METADATA': '{"question_id": 449286255.0, "answer_id": 1780190741.0, "url": "https://www.zhihu.com/question/449286255/answer/1780190741", "upvotes": "赞同", "answer_creation_time": "2021-03-14T16:09:26.000Z"}'},
           {'INSTRUCTION': '灯笼是一种在中国被广泛认可的一种吉祥物件',
           'RESPONSE': '灯笼又统称为灯彩，是一种古老的汉族传统工艺品。经过数千年的发展，灯笼发展出了不同的地域风格，每一种灯笼都具有独特的艺术表现形式。每年的农历正月十五元宵节前后，人们都挂起象征团圆意义的红灯笼，来营造一种喜庆的氛围。',
           'SOURCE': 'Zhihu',
           'METADATA': '{"question_id": 449286255.0, "answer_id": 1780190741.0, "url": '
                       '"https://www.zhihu.com/question/449286255/answer/1780190741", "upvotes": "赞同", '
                       '"answer_creation_time": "2021-03-14T16:09:26.000Z"}'},
           {'INSTRUCTION': '中国是如何实习四个现代化?',
           'RESPONSE': '中国的四个现代化最早由中国共产党提出，在经过了几代人的努力及中国共产党的领导下，一步一个脚印扎实工作，努力奋斗，最终在全国人民的团结努力下实现的。',
           'SOURCE': 'Zhihu',
           'METADATA': '{"question_id": 449286255.0, "answer_id": 1780190741.0, "url": "https://www.zhihu.com/question/449286255/answer/1780190741", "upvotes": "赞同", "answer_creation_time": "2021-03-14T16:09:26.000Z"}'
           }]

questions = []
for data in corpus:
    questions.append(data['INSTRUCTION'])
# print(corpus)
corpus_embeddings = embedder.encode(questions, convert_to_tensor=True)

# Query sentences:
queries = ["基因测序准确性怎么样"]


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)

    # print("Result:",questions[613248])
    # print("Result:", questions[613249])
    # print("Result:", questions[613247])
    # print("Result:", corpus_embeddings[613248])
    print("Reuslt:",questions[top_results[1][0]],"score:",top_results[0][0])
    print("Reuslt:", corpus_embeddings[top_results[1][0]], "score:", top_results[0][0])
    # print("Reuslt:", top_results[1][0], "score:", top_results[0][0])
    # print("top_result:",top_results)
    # print("\nTop 5 most similar sentences in corpus:")
    # print(top_results)
    # for score, idx in zip(top_results[0], top_results[1]):
    #     print(corpus[idx], "(Score: {:.4f})".format(score))

    """
    # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
    """

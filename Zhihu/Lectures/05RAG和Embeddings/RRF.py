# 1. 基于关键字检索的排序

import time


class MyEsConnector:
    def __init__(self, es_client, index_name, keyword_fn):
        self.es_client = es_client
        self.index_name = index_name
        self.keyword_fn = keyword_fn

    def add_documents(self, documents):
        """文档灌库"""
        if self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.delete(index=self.index_name)
        self.es_client.indices.create(index=self.index_name)
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "keywords": self.keyword_fn(doc),
                    "text": doc,
                    "id": f"doc_{i}",
                },
            }
            for i, doc in enumerate(documents)
        ]
        helpers.bulk(self.es_client, actions)
        time.sleep(1)

    def search(self, query_string, top_n=3):
        """检索"""
        search_query = {"match": {"keywords": self.keyword_fn(query_string)}}
        res = self.es_client.search(
            index=self.index_name, query=search_query, size=top_n
        )
        return {
            hit["_source"]["id"]: {
                "text": hit["_source"]["text"],
                "rank": i,
            }
            for i, hit in enumerate(res["hits"]["hits"])
        }


from chinese_utils import to_keywords  # 使用中文的关键字提取函数

es = Elasticsearch(
    hosts=["http://117.50.198.53:9200"],  # 服务地址与端口
    http_auth=("elastic", "FKaB1Jpz0Rlw0l6G"),  # 用户名，密码
)

# 创建 ES 连接器
es_connector = MyEsConnector(es, "demo_es_rrf", to_keywords)

# 文档灌库
es_connector.add_documents(documents)

# 关键字检索
keyword_search_results = es_connector.search(query, 3)

print(keyword_search_results)

# {'doc_2': {'text': '张某经诊断为非小细胞肺癌III期', 'rank': 0}, 'doc_0': {'text': '李某患有肺癌，癌细胞已转移', 'rank': 1}, 'doc_3': {'text': '小细胞肺癌是肺癌的一种', 'rank': 2}}

# 2. 基于向量检索的排序

# 创建向量数据库连接器
vecdb_connector = MyVectorDBConnector("demo_vec_rrf", get_embeddings)

# 文档灌库
vecdb_connector.add_documents(documents)

# 向量检索
vector_search_results = {
    "doc_" + str(documents.index(doc)): {"text": doc, "rank": i}
    for i, doc in enumerate(vecdb_connector.search(query, 3)["documents"][0])
}  # 把结果转成跟上面关键字检索结果一样的格式

print(vector_search_results)

# {'doc_3': {'text': '小细胞肺癌是肺癌的一种', 'rank': 0}, 'doc_0': {'text': '李某患有肺癌，癌细胞已转移', 'rank': 1}, 'doc_2': {'text': '张某经诊断为非小细胞肺癌III期', 'rank': 2}}

# 3. 基于 RRF 的融合排序


def rrf(ranks, k=1):
    ret = {}
    # 遍历每次的排序结果
    for rank in ranks:
        # 遍历排序中每个元素
        for id, val in rank.items():
            if id not in ret:
                ret[id] = {"score": 0, "text": val["text"]}
            # 计算 RRF 得分
            ret[id]["score"] += 1.0 / (k + val["rank"])
    # 按 RRF 得分排序，并返回
    return dict(sorted(ret.items(), key=lambda item: item[1]["score"], reverse=True))


import json

# 融合两次检索的排序结果
reranked = rrf([keyword_search_results, vector_search_results])

print(json.dumps(reranked, indent=4, ensure_ascii=False))

# {
#     "doc_2": {
#         "score": 1.3333333333333333,
#         "text": "张某经诊断为非小细胞肺癌III期"
#     },
#     "doc_3": {
#         "score": 1.3333333333333333,
#         "text": "小细胞肺癌是肺癌的一种"
#     },
#     "doc_0": {
#         "score": 1.0,
#         "text": "李某患有肺癌，癌细胞已转移"
#     }
# }

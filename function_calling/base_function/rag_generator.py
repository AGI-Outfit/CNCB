from function_calling.base_function.base_tool import BaseTool

import faiss
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np


class RAGGenerator(BaseTool):
    def __init__(self, args: dict):
        """
        用于存储RAG的参数，匹配文件
        """
        super().__init__(args)
        # RAG参数初始化
        if self.args_rag["use_rag"]:
            self.docs = self.args_rag["docs"]

            self.tokenizer = AutoTokenizer.from_pretrained(self.args_rag["model_path"])
            # tokenizer.save_pretrained("/home/jk/")
            self.model = AutoModel.from_pretrained(self.args_rag["model_path"])
            # 文本转换为向量
            tokenized_docs = [self.tokenizer.encode(doc, return_tensors='pt') for doc in self.docs]
            # 使用模型获取文本的embedding
            with torch.no_grad():
                embeddings = [self.model(text)[0].mean(dim=1).numpy() for text in tokenized_docs]
            # 将嵌入向量转换为FAISS所需的格式，并构建索引
            embeddings = np.squeeze(embeddings)
            d = embeddings[0].shape[0]  # 维度
            self.index = faiss.IndexFlatL2(d)  # 使用L2距离作为度量标准，创建索引
            # 转换为FAISS接受的float32数组格式
            emb_array = np.array(embeddings, dtype='float32')
            # 添加向量到索引中
            self.index.add(emb_array)
            self.level_print("Success: RAG Init.", 1)

    def __call__(self, query: str, k: int = 2) -> str:
        if self.args_rag["use_rag"]:
            # 现在你可以使用这个索引来搜索最相似的向量了
            query_tokenized = self.tokenizer.encode(query, return_tensors='pt')
            with torch.no_grad():
                query_emb = self.model(query_tokenized)[0].mean(dim=1).numpy()
            query_emb = np.squeeze(query_emb)
            # 转换查询向量格式
            query_emb = np.array([query_emb], dtype='float32')

            # 搜索最相似的k个向量
            distances, indices = self.index.search(query_emb, k)
            str_to_print = ""
            for i in range(k):
                str_to_print += f"Similar sentence #{i + 1} with distance {distances[0][i]}: " + self.docs[indices[0][i]] + "\n"
            self.level_print("RAG Content:\n" + str_to_print, 3)
            return self.docs[indices[0][0]]
        else:
            return ""

    # def _retriever(self, query: str) -> List[Document]:
    #     """
    #     这个函数目前没有使用
    #     功能是根据匹配分数输出匹配文本并输出分数
    #     :param query:
    #     :return:
    #     """
    #     docs, scores = zip(*(self.vectorstore.similarity_search_with_score(query, k=5)))
    #     for doc, score in zip(docs, scores):
    #         doc.metadata["score"] = score
    #     return docs
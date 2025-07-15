# 使用之前请将第57行 DEEPSEEK_API_KEY 替换为自己的api
import os
import ast
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 使用Transformers 库的重排器封装类
class TransformersReranker:
    def __init__(self, model_name_or_path: str, device: str = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():  # for Apple Silicon
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        print(f"正在从 '{model_name_or_path}' 加载 Reranker 模型到设备 '{self.device}'...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
            self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            print("Reranker 模型加载成功。")
        except Exception as e:
            raise IOError(f"加载 Reranker 模型失败: {e}")

    def compute_score(self, sentence_pairs: list[list[str]]) -> list[float]:
        if not sentence_pairs:
            return []

        # 禁用梯度计算，加速并减少内存使用
        with torch.no_grad():
            inputs = self.tokenizer(sentence_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs_on_device = {key: val.to(self.device) for key, val in inputs.items()}

            # 模型推理
            scores = self.model(**inputs_on_device, return_dict=True).logits.view(-1).float()

            # 将分数从 PyTorch Tensor 转换为 Python 列表
            return scores.cpu().tolist()


DATA_DIR = "./legal_data"
EMBEDDING_MODEL_NAME = "F:/My_model_cache/hub/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620"
RERANKER_MODEL_NAME = 'F:/My_model_cache\hub/models--BAAI--bge-reranker-base/snapshots/2cfc18c9415c912f9d8155881c133215df768a70'
VECTOR_DB_PATH = "./vector_store/faiss_index_legal"
DEEPSEEK_API_KEY = "use your api key"


try:
    reranker = TransformersReranker(RERANKER_MODEL_NAME)
except Exception as e:
    reranker = None
    print(f"警告：初始化 TransformersReranker 失败: {e}。将跳过重排步骤。")

# 中文法律条纹分割器，按章、节、条等进行分割
import re
from typing import List
from langchain.text_splitter import TextSplitter

class ChineseLegalTextSplitter(TextSplitter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.split_pattern = (
            r'(?m)^\s*第[一二三四五六七八九十百千万\d]+章.*|'
            r'(?m)^\s*第[一二三四五六七八九十百千万\d]+节.*|'
            r'(?m)^\s*第[一二三四五六七八九十百千万\d]+条.*|'
            r'（[一二三四五六七八九十百千万\d]+）|'
            r'\([一二三四五六七八九十百千万\d]+\)|'
            r'[一二三四五六七八九十百千万\d]+、'
        )

    def split_text(self, text: str) -> List[str]:
        # 匹配到分割符的位置，分割符留在下一段文本的开头。
        splits = re.split(f'({self.split_pattern})', text)
        # 将分隔符和它后面的文本合并
        chunks = []
        current_chunk = ""
        for i, part in enumerate(splits):
            if i % 2 == 0:  # 偶数索引是文本
                current_chunk += part
            else:  # 奇数索引是分隔符
                if current_chunk.strip():  # 如果当前块有内容，先存起来
                    chunks.append(current_chunk.strip())
                current_chunk = part  # 分隔符作为新块的开始

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # 后处理，过滤
        final_chunks = self._filter_short_and_irrelevant_chunks(chunks)
        return final_chunks

    # 过滤太短或无关键词的块
    def _filter_short_and_irrelevant_chunks(self, chunks: List[str], min_length: int = 30) -> List[str]:
        filtered_chunks = []
        irrelevant_keywords = ["Table of Contents", "目\s*录"]  # 匹配 "目录" 或 "目 录"
        for chunk in chunks:
            if len(chunk) < min_length:
                continue
            if any(re.search(keyword, chunk, re.IGNORECASE) for keyword in irrelevant_keywords):
                continue
            filtered_chunks.append(chunk)
        return filtered_chunks


# --- 核心功能 ---

def initialize_vector_db(force_recreate=False):
    print("正在初始化或加载向量数据库...")
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs=model_kwargs,
                                       encode_kwargs=encode_kwargs)

    if os.path.exists(VECTOR_DB_PATH) and not force_recreate:
        print("从缓存加载知识库成功！")
        vector_db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("缓存不存在或强制重新创建，开始构建新的知识库...")
        if not os.path.exists(DATA_DIR):
            print(f"错误：未找到数据目录：{DATA_DIR}")
            return None

        all_documents = []
        all_files = os.listdir(DATA_DIR)

        print(f"在 {DATA_DIR} 发现 {len(all_files)} 个文件，开始处理...")
        for file_name in all_files:
            file_path = os.path.join(DATA_DIR, file_name)
            if file_name.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                all_documents.extend(loader.load())
            elif file_name.lower().endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                all_documents.extend(loader.load())

        if not all_documents:
            print("未能从支持的文件中提取任何内容。")
            return None

        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        text_splitter = ChineseLegalTextSplitter() # 改为使用中文法律条纹分割器
        chunks = text_splitter.split_documents(all_documents)

        # print一些分割后的示例，用来调试
        print(f"文档分割完成，共得到 {len(chunks)} 个语义块。")
        print("前3个块的示例:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"--- 块 {i + 1} (长度: {len(chunk.page_content)}) ---")
            print(chunk.page_content[:200] + "...")  # 打印前200个字符
            print("\n")

        print(f"正在为 {len(chunks)} 个文本块创建向量索引...")
        vector_db = FAISS.from_documents(chunks, embeddings)

        os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
        vector_db.save_local(VECTOR_DB_PATH)
        print("知识库初始化并缓存成功！")

    return vector_db


def _expand_query(query: str, client: OpenAI) -> list[str]:
    prompt = f"""你是一个专业的查询分析和扩展助手。请根据用户的原始问题，生成3个不同角度但语义高度相关的查询，用于在法律知识库中进行更全面的搜索。请确保生成的查询是具体、清晰的，并且直接可用作搜索词。

返回一个 Python 列表格式的字符串，例如：["查询1", "查询2", "查询3"]。

原始问题: "{query}" """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # 或者其他你选择的模型
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # 较低的温度确保相关性
            max_tokens=150
        )
        expanded_queries_str = response.choices[0].message.content

        try:
            queries = ast.literal_eval(expanded_queries_str)
            if isinstance(queries, list):
                # 将原始查询添加到列表头部，并去重
                all_queries = [query] + queries
                return list(dict.fromkeys(all_queries))
        except (ValueError, SyntaxError):
            # 此时解析失败，v报错 只返回原始查询
            print(f"警告：解析LLM生成的查询扩展结果失败。结果：{expanded_queries_str}")
            pass

    except Exception as e:
        print(f"警告：调用LLM进行查询扩展时发生错误: {e}")

    # 发生任何错误或异常时，都只返回原始查询
    return [query]

# 集成查询扩展 重排 元数据溯源
def get_llm_response(query: str, vector_db: FAISS):
    if not DEEPSEEK_API_KEY:
        return "错误：请配置您的 API Key。"

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

    # 查询扩展
    print(f"原始查询: {query}")
    all_queries = _expand_query(query, client)
    print(f"扩展后查询: {all_queries}")

    # 召回
    all_retrieved_docs = []
    for q in all_queries:
        # 每个扩展查询召回 k 个文档
        retrieved_docs = vector_db.similarity_search(q, k=5)
        all_retrieved_docs.extend(retrieved_docs)

    # 对所有召回的文档进行去重，保持文档的唯一性
    unique_docs = {doc.page_content: doc for doc in all_retrieved_docs}.values()

    if not unique_docs:
        return f"根据我所掌握的资料，无法回答您关于“{query}”的问题，请您咨询专业律师。"

    # 重排
    if reranker:
        print(f"召回了 {len(unique_docs)} 个独立文档，正在进行重排...")
        pairs = [[query, doc.page_content] for doc in unique_docs]

        # 使用新的 TransformersReranker 实例
        scores = reranker.compute_score(pairs)
        doc_with_scores = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)

        # 选择重排后分数最高的 top_k 个文档
        final_top_k = 3
        final_docs = [doc for doc, score in doc_with_scores[:final_top_k]]

        # 调试，查看重排后的结果
        print("--- Reranked Docs (Top 3) ---")
        for doc, score in doc_with_scores[:final_top_k]:
            source = doc.metadata.get('source', '未知来源').split('/')[-1]  # 只取文件名
            print(f"Score: {score:.4f} | Source: {source} | Content: {doc.page_content[:80]}...")
    else:
        print("跳过重排步骤。")
        final_docs = list(unique_docs)[:3]

    # 构建上下文并追溯来源
    context_str = "\n\n".join([doc.page_content for doc in final_docs])

    # 从元数据中提取来源信息
    sources = [doc.metadata.get('source', '未知来源') for doc in final_docs]

    # 只保留文件名并去重
    unique_source_files = list(dict.fromkeys([os.path.basename(s) for s in sources]))


    # 调用 LLM 生成回答
    system_prompt = (
        "你是一个专业的法律问答助手。请根据下面提供的【参考资料】，用中文自信、准确、简洁、专业、严谨地回答用户的【问题】。你要耐心，鼓励用户提问。"
        "你的回答应该严格基于【参考资料】内容，避免提供个人观点或猜测，并应富有同情心。"
        "如果【参考资料】中没有足够的信息来回答问题，请只回答：“根据我所掌握的资料，无法回答您关于‘{query}’的问题，请您咨询专业律师。”，不要添加任何其他内容。"
    )

    user_prompt_content = f"【参考资料】\n{context_str}\n\n【问题】\n{query}"
    messages = [{"role": "system", "content": system_prompt.format(query=query)},
                {"role": "user", "content": user_prompt_content}]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.1  # 回答阶段使用极低的温度，确保严谨性
        )
        response_text = response.choices[0].message.content

        # 免责声明
        disclaimer = (
            f"\n\n---\n"
            f"*请注意，以上信息仅为基于本地法律文档的参考，不构成正式的法律建议。 "
            f"内容主要参考自：{', '.join(unique_source_files)}。如有需要，请咨询专业律师。*"
        )

        # 只有在LLM能回答问题时，才附上免责声明和来源
        if not "无法回答" in response_text:
            return response_text + disclaimer
        else:
            return response_text

    except Exception as e:
        return f"调用大语言模型 API 时发生错误: {e}"


# 加载向量数据库，避免每次请求都重新加载
VECTOR_DB_INSTANCE = initialize_vector_db()

# FastAPI调用
def query_rag_system(query: str):
    if VECTOR_DB_INSTANCE is None:
        return "错误：知识库未能成功初始化，请检查后台日志。"
    return get_llm_response(query, VECTOR_DB_INSTANCE)


# 测试RAG功能 强制重建索引
# 测试RAG功能/重建索引
if __name__ == '__main__':
    print("--- 正在进行 RAG 核心模块测试 ---")

    # 1. 强制重新创建数据库并获取新实例
    print("\n[步骤 1: 强制重新创建知识库]")
    test_db_instance = initialize_vector_db(force_recreate=True)

    if test_db_instance:
        # 2. 定义测试问题
        test_query = "劳动合同需要包含哪些内容？"
        print(f"\n[步骤 2: 测试问题]\n\"{test_query}\"")

        # 3. 直接使用新创建的实例进行测试
        print("\n[步骤 3: 获取 LLM 回答]")
        response = get_llm_response(test_query, test_db_instance)

        # 4. 打印结果
        print("\n[步骤 4: 测试回答]")
        print("-" * 20)
        print(response)
        print("-" * 20)
    else:
        print("\n测试失败：知识库未能成功初始化。")

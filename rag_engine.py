"""
finance_rag_engine.py
金融投研RAG系统 - 核心引擎
包含：文档加载、文本切分、向量嵌入、FAISS存储、检索与生成
"""

import os
import json
import re
import time
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer

# ====== 配置 ======
CHUNK_SIZE = 400       # 每段文字最大字符数
CHUNK_OVERLAP = 80     # 段间重叠字符数
TOP_K = 5              # 检索返回的段落数

DATA_DIR = Path(__file__).parent / "data"
INDEX_DIR = Path(__file__).parent / "index"

# 自动检测运行环境：本地有缓存则离线加载，云端则在线下载
_HF_CACHE = Path.home() / ".cache/huggingface/hub"
_MODEL_SNAPSHOT = _HF_CACHE / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
if _MODEL_SNAPSHOT.exists():
    # 本地环境：直接使用缓存，无需网络
    EMBED_MODEL = str(_MODEL_SNAPSHOT)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
else:
    # 云端/初次运行：从 HuggingFace 下载模型
    EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ====== 全局单例 ======
_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.IndexFlatIP] = None
_chunks: List[Dict] = []   # [{"text": ..., "source": ..., "chunk_id": ...}]


def get_model() -> SentenceTransformer:
    """懒加载嵌入模型（单例）"""
    global _model
    if _model is None:
        print("[RAG] 加载嵌入模型...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


# ====== 1. 文档加载 ======
def load_documents(data_dir: Path) -> List[Dict]:
    """加载 data/ 目录下所有 .txt 文件"""
    docs = []
    for fpath in sorted(data_dir.glob("*.txt")):
        text = fpath.read_text(encoding="utf-8", errors="ignore")
        # 基础清洗：去除多余空行
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        docs.append({"filename": fpath.name, "text": text})
    print(f"[RAG] 共加载 {len(docs)} 份文档")
    return docs


# ====== 2. 文本切分 ======
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    按段落优先 + 滑动窗口的混合策略切分文本
    1. 先按双换行切成自然段落
    2. 超长段落再按 chunk_size 切割（带 overlap）
    """
    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    buf = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) + 2 <= chunk_size:
            buf = (buf + "\n\n" + para).strip() if buf else para
        else:
            if buf:
                chunks.append(buf)
            # 如果单段落超长，滑动切割
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    sub = para[i: i + chunk_size].strip()
                    if sub:
                        chunks.append(sub)
                buf = ""
            else:
                # overlap：用上一个 chunk 末尾做上下文
                if chunks:
                    tail = chunks[-1][-overlap:] if len(chunks[-1]) > overlap else chunks[-1]
                    buf = (tail + "\n\n" + para).strip()
                else:
                    buf = para
    if buf:
        chunks.append(buf)
    return chunks


def build_chunks(docs: List[Dict]) -> List[Dict]:
    """对所有文档做切分，返回带来源信息的 chunk 列表"""
    all_chunks = []
    for doc in docs:
        text_chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "text": chunk,
                "source": doc["filename"],
                "chunk_id": f"{doc['filename']}_{i}"
            })
    print(f"[RAG] 共生成 {len(all_chunks)} 个文本片段")
    return all_chunks


# ====== 3. 向量嵌入 & FAISS 索引 ======
def build_index(chunks: List[Dict], index_dir: Path) -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    """
    为 chunks 生成嵌入向量，构建 FAISS 内积索引（等效余弦相似度）
    将索引和 chunk 元数据持久化到 index_dir
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = index_dir / "finance.index"
    meta_path = index_dir / "chunks.json"

    # 如果已有缓存则直接加载
    if faiss_path.exists() and meta_path.exists():
        print("[RAG] 加载已缓存的向量索引...")
        index = faiss.read_index(str(faiss_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            loaded_chunks = json.load(f)
        return index, loaded_chunks

    model = get_model()
    print("[RAG] 生成嵌入向量（首次运行可能需要1-3分钟）...")
    texts = [c["text"] for c in chunks]
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    print(f"[RAG] 嵌入完成，耗时 {time.time()-t0:.1f}s")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # 内积（归一化后=余弦相似度）
    index.add(embeddings.astype("float32"))

    # 持久化
    faiss.write_index(index, str(faiss_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[RAG] 索引构建完成，共 {index.ntotal} 个向量")
    return index, chunks


def load_or_build_index() -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    """全局：加载或构建索引，更新全局变量"""
    global _index, _chunks
    if _index is not None:
        return _index, _chunks
    docs = load_documents(DATA_DIR)
    chunks = build_chunks(docs)
    _index, _chunks = build_index(chunks, INDEX_DIR)
    return _index, _chunks


# ====== 4. 检索 ======
def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    """
    检索与问题最相关的 top_k 个文本片段
    返回带 score、source、text 的字典列表
    """
    index, chunks = load_or_build_index()
    model = get_model()
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_vec, top_k)
    results = []
    seen_sources = {}  # 去重：每个来源最多保留 2 个片段
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        src = chunk["source"]
        if seen_sources.get(src, 0) < 2:
            results.append({
                "text": chunk["text"],
                "source": src,
                "chunk_id": chunk["chunk_id"],
                "score": float(score)
            })
            seen_sources[src] = seen_sources.get(src, 0) + 1
    return results


# ====== 5. Prompt 构建 ======
SYSTEM_PROMPT = """你是一名资深金融分析师，专注于A股上市公司和中国资本市场研究。
你的回答应当：
1. 严格基于提供的参考资料，不得捏造数据
2. 回答专业、客观，语言清晰
3. 每个关键数据点必须注明来源文档
4. 如涉及风险，主动提示投资风险
5. 若参考资料中无相关信息，明确告知"知识库中暂无该信息"

⚠️ 风险提示：本系统仅供学习研究，不构成任何投资建议。"""


def build_prompt(query: str, retrieved: List[Dict]) -> Tuple[str, str]:
    """
    将检索到的片段组装成 Prompt
    返回 (system_message, user_message)
    """
    ctx_parts = []
    for i, item in enumerate(retrieved, 1):
        src_name = item["source"].replace(".txt", "").replace("_", " ")
        ctx_parts.append(f"【参考资料{i}】来源：{src_name}\n{item['text']}")
    context = "\n\n---\n\n".join(ctx_parts)

    user_msg = f"""请根据以下参考资料回答问题。

{context}

---

用户问题：{query}

请基于上述资料给出专业分析回答，并在回答末尾列出所引用的资料来源。"""
    return SYSTEM_PROMPT, user_msg


# ====== 6. LLM 调用（支持多种 API）======
def call_llm(system_prompt: str, user_prompt: str, api_key: str = "",
             base_url: str = "", model_name: str = "",
             stream: bool = False):
    """
    调用 LLM API 生成回答
    支持：DeepSeek / OpenAI 兼容接口 / SiliconFlow
    stream=True 时返回生成器，否则返回完整字符串
    """
    from openai import OpenAI, APIError, AuthenticationError

    if not api_key:
        # 没有 API Key，直接返回模拟回答（用于测试）
        return _mock_answer(user_prompt)

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    final_model = model_name or "deepseek-chat"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        if stream:
            def _gen():
                response = client.chat.completions.create(
                    model=final_model,
                    messages=messages,
                    stream=True,
                    temperature=0.3,
                    max_tokens=2048
                )
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
            return _gen()
        else:
            response = client.chat.completions.create(
                model=final_model,
                messages=messages,
                stream=False,
                temperature=0.3,
                max_tokens=2048
            )
            return response.choices[0].message.content

    except AuthenticationError as e:
        msg = str(e)
        if "401" in msg or "authentication" in msg.lower():
            raise RuntimeError(
                "❌ API Key 认证失败（401）\n\n"
                "可能原因：\n"
                "1. API Key 填写错误或已过期\n"
                "2. 在 DeepSeek 平台重新生成一个新的 Key\n"
                "3. 检查 Key 是否从 platform.deepseek.com 的【API Keys】页面获取"
            ) from e
        raise

    except Exception as e:
        msg = str(e)
        if "402" in msg or "Insufficient Balance" in msg:
            raise RuntimeError(
                "❌ 账户余额不足（402 - Insufficient Balance）\n\n"
                "DeepSeek API 需要账户有余额才能使用。\n\n"
                "💡 三种解决方案：\n"
                "1️⃣ 充值 DeepSeek：登录 platform.deepseek.com → 充值（最低 5 元）\n"
                "2️⃣ 免费使用 SiliconFlow：注册 cloud.siliconflow.cn → 获取 Key\n"
                "   （新用户送 14 元，Qwen2.5 模型 ¥1.33/百万 tokens）\n"
                "3️⃣ 清空 API Key → 点「保存配置」→ 使用演示模式直接看检索结果\n"
                "   （同样能展示 RAG 完整流程，可以正常交作业）"
            ) from e
        if "403" in msg or "forbidden" in msg.lower():
            raise RuntimeError(
                "❌ 访问被拒绝（403）\n\n"
                "可能原因：\n"
                "1. API Key 没有调用该模型的权限\n"
                "2. 账户被限制，请联系对应平台客服"
            ) from e
        if "429" in msg or "rate" in msg.lower():
            raise RuntimeError(
                "❌ 请求频率过高（429）\n\n"
                "请稍等 1-2 分钟后重试，或降低提问频率。"
            ) from e
        raise


def _mock_answer(user_prompt: str) -> str:
    """
    未配置 API Key 时的模拟回答（演示用）
    基于检索到的内容做简单提取，不调用外部 LLM
    """
    # 提取参考资料部分
    lines = user_prompt.split("\n")
    answer_lines = ["**[演示模式 - 未配置 API Key，以下为基于知识库直接提取的内容]**\n"]
    in_ref = False
    ref_texts = []
    for line in lines:
        if line.startswith("【参考资料"):
            in_ref = True
            ref_texts.append(line)
        elif line.strip() == "---":
            in_ref = False
        elif in_ref:
            ref_texts.append(line)
    if ref_texts:
        answer_lines.append("根据知识库中检索到的相关内容：\n")
        answer_lines.extend(ref_texts[:30])  # 最多展示30行
        answer_lines.append("\n\n---\n⚠️ **风险提示**：本系统仅供学习研究，不构成任何投资建议。")
    return "\n".join(answer_lines)


# ====== 7. 完整 RAG 问答流程 ======
def rag_answer(query: str, api_key: str = "", base_url: str = "",
               model_name: str = "", top_k: int = TOP_K,
               stream: bool = False):
    """
    完整 RAG 流程：检索 → 构建 Prompt → 生成回答
    返回 (answer_str_or_generator, retrieved_chunks)
    """
    # 1. 检索
    retrieved = retrieve(query, top_k=top_k)
    if not retrieved:
        return "抱歉，知识库中未找到与该问题相关的内容，请换一种问法。", []

    # 2. 构建 Prompt
    sys_prompt, user_prompt = build_prompt(query, retrieved)

    # 3. 调用 LLM
    answer = call_llm(sys_prompt, user_prompt, api_key=api_key,
                      base_url=base_url, model_name=model_name, stream=stream)

    return answer, retrieved


# ====== 测试入口 ======
if __name__ == "__main__":
    print("=" * 60)
    print("金融投研 RAG 系统 - 核心引擎测试")
    print("=" * 60)

    # 构建索引
    load_or_build_index()

    # 测试检索
    test_queries = [
        "比亚迪2023年上半年的汽车毛利率是多少？",
        "宁德时代2023年全年净利润和动力电池市占率",
        "贵州茅台的分红比例和股息率",
        "招商银行的净息差是多少？不良贷款率如何？",
        "理想汽车2023年交付量和净利润情况",
    ]

    for q in test_queries:
        print(f"\n{'='*50}")
        print(f"问题：{q}")
        retrieved = retrieve(q, top_k=3)
        print(f"检索到 {len(retrieved)} 个相关片段：")
        for r in retrieved:
            print(f"  - [{r['source']}] 相似度={r['score']:.3f}")
            print(f"    {r['text'][:100]}...")

    print("\n✅ 引擎测试完成！")

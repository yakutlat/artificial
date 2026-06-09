"""
app.py
金融投研 RAG 智能问答系统 - Streamlit 前端
"""

import streamlit as st
import sys
import os
import time
from pathlib import Path

# 确保能找到 rag_engine
sys.path.insert(0, str(Path(__file__).parent))

# ====== 页面配置 ======
st.set_page_config(
    page_title="金融投研 RAG 智能问答系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== 自定义 CSS ======
st.markdown("""
<style>
/* 主色调 */
:root {
    --primary: #1a56db;
    --bg-card: #f8faff;
}

/* 聊天消息气泡 */
.user-bubble {
    background: #1a56db;
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 85%;
    margin-left: auto;
    word-break: break-word;
}
.assistant-bubble {
    background: #f0f4ff;
    color: #1a1a2e;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 90%;
    border-left: 3px solid #1a56db;
    word-break: break-word;
}

/* 来源引用卡片 */
.source-card {
    background: #fff8e1;
    border: 1px solid #ffd54f;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.85em;
}

/* 侧边栏 */
.sidebar-info {
    background: #e8f4fd;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
    font-size: 0.85em;
}

/* 指标卡片 */
.metric-chip {
    display: inline-block;
    background: #e8f0fe;
    color: #1a56db;
    border-radius: 12px;
    padding: 3px 10px;
    font-size: 0.8em;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ====== 初始化 Session State ======
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "base_url" not in st.session_state:
    st.session_state.base_url = "https://api.deepseek.com"
if "model_name" not in st.session_state:
    st.session_state.model_name = "deepseek-chat"
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "show_sources" not in st.session_state:
    st.session_state.show_sources = True


# ====== 加载 RAG 引擎 ======
@st.cache_resource(show_spinner=False)
def load_rag_engine():
    """加载并初始化 RAG 引擎（带缓存，只运行一次）"""
    from rag_engine import load_or_build_index, get_model
    get_model()  # 预热模型
    index, chunks = load_or_build_index()
    return index, chunks


# ====== 侧边栏 ======
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/combo-chart--v2.png", width=48)
    st.title("⚙️ 系统配置")

    st.markdown("---")
    st.subheader("🔑 LLM 接入配置")

    st.info("💡 **推荐**：使用 **SiliconFlow**（硅基流动），新用户送 14 元免费额度！\n注册地址：cloud.siliconflow.cn\n⚠️ 7B模型可能产生数字偏差，建议选 **Qwen2.5-72B-Instruct** 以获得最佳准确度")

    api_key_input = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-xxx（不填则使用演示模式）",
        help="支持 DeepSeek / OpenAI 兼容接口。不填时系统以演示模式运行，直接展示检索到的原文内容。"
    )

    base_url_input = st.selectbox(
        "API 接口",
        options=[
            "https://api.siliconflow.cn/v1",
            "https://api.deepseek.com",
            "https://api.openai.com/v1",
        ],
        index=0,
        help="SiliconFlow 新用户免费 14 元，DeepSeek 最低充值 5 元"
    )

    price_hints = {
        "https://api.siliconflow.cn/v1": "🆓 新用户送 14 元 · Qwen2.5 仅 ¥1.33/百万tokens",
        "https://api.deepseek.com": "💰 需充值 · 最低 5 元 · deepseek-chat ¥1/百万tokens",
        "https://api.openai.com/v1": "💲 国际信用卡 · GPT-4o-mini $0.15/百万tokens",
    }
    st.caption(price_hints.get(base_url_input, ""))

    model_options = {
        "https://api.siliconflow.cn/v1": ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-V2.5"],
        "https://api.deepseek.com": ["deepseek-chat", "deepseek-reasoner"],
        "https://api.openai.com/v1": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    }
    model_list = model_options.get(base_url_input, ["deepseek-chat"])
    model_name_input = st.selectbox("模型", options=model_list)

    if st.button("💾 保存配置", use_container_width=True):
        st.session_state.api_key = api_key_input
        st.session_state.base_url = base_url_input
        st.session_state.model_name = model_name_input
        st.success("✅ 配置已保存！现在可以开始提问了。")

    st.markdown("---")
    st.subheader("🔍 检索设置")
    top_k = st.slider("检索片段数（Top-K）", min_value=1, max_value=10, value=5)
    show_sources = st.toggle("显示参考来源", value=True)
    st.session_state.top_k = top_k
    st.session_state.show_sources = show_sources

    st.markdown("---")
    st.subheader("📚 知识库概况")

    data_dir = Path(__file__).parent / "data"
    doc_files = list(data_dir.glob("*.txt"))
    st.markdown(f"""
    <div class="sidebar-info">
    📄 文档数量：<strong>{len(doc_files)}</strong> 份<br>
    🏢 覆盖领域：新能源 · 白酒 · 银行 · 科技 · 消费 · 医疗 · 能源 · 半导体<br>
    📅 数据区间：2022-2024年<br>
    🔧 嵌入模型：multilingual-MiniLM<br>
    🗄️ 向量库：FAISS（内积检索）
    </div>
    """, unsafe_allow_html=True)

    # 文档列表
    with st.expander("查看全部文档"):
        for f in sorted(doc_files):
            name = f.name.replace(".txt", "").replace("_", " ")
            st.markdown(f"- 📑 {name}")

    st.markdown("---")
    if st.button("🗑️ 清空对话历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("⚠️ 本系统仅供学习研究\n不构成任何投资建议")


# ====== 主区域 ======
st.title("📊 金融投研 RAG 智能问答系统")
st.caption("基于 RAG（检索增强生成）技术 · 覆盖30份权威财报与行业研报")

# 初始化索引（带进度提示）
if not st.session_state.index_ready:
    with st.spinner("⏳ 正在初始化向量知识库（首次加载约需1-2分钟）..."):
        try:
            load_rag_engine()
            st.session_state.index_ready = True
        except Exception as e:
            st.error(f"❌ 知识库初始化失败：{e}")
            st.stop()
    st.success("✅ 知识库就绪！现在可以开始提问了。")
    time.sleep(0.8)
    st.rerun()

# 状态提示栏
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("知识库文档", "30份", "↑较基础要求+200%")
with col2:
    mode = "LLM模式" if st.session_state.api_key else "演示模式"
    st.metric("运行模式", mode)
with col3:
    st.metric("检索策略", f"Top-{st.session_state.top_k}")
with col4:
    st.metric("向量模型", "多语言MiniLM")

st.markdown("---")

# 示例问题
if not st.session_state.messages:
    st.subheader("💡 示例问题（点击即可提问）")
    example_questions = [
        "比亚迪2023年上半年的汽车毛利率是多少？相比去年同期表现如何？",
        "宁德时代2023年全年净利润是多少？动力电池全球市占率如何？",
        "贵州茅台和五粮液2023年净利润、毛利率各是多少？两者有何差异？",
        "招商银行2023年的净息差是多少？与工商银行相比如何？",
        "中国新能源汽车行业2023年总销量和渗透率是多少？",
        "理想汽车2023年交付量和净利润，与蔚来、小鹏相比如何？",
        "拼多多2023年营收增速为什么这么高？Temu业务表现如何？",
        "中国光伏行业2023年组件价格跌了多少？主要原因是什么？",
        "中国平安2023年寿险新业务价值（NBV）增长了多少？",
        "美的集团和格力电器2023年营收、利润有什么差异？",
    ]
    cols = st.columns(2)
    for i, q in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(f"💬 {q}", key=f"eg_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()


# ====== 渲染历史对话 ======
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-bubble">
        🙋 {msg["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-bubble">
        {msg["content"]}
        </div>
        """, unsafe_allow_html=True)
        # 显示来源
        if st.session_state.show_sources and "sources" in msg and msg["sources"]:
            with st.expander(f"📎 参考来源（{len(msg['sources'])}个片段）"):
                for j, src in enumerate(msg["sources"], 1):
                    src_name = src["source"].replace(".txt", "").replace("_", " ")
                    st.markdown(f"""
                    <div class="source-card">
                    <strong>📄 [{j}] {src_name}</strong>
                    <span class="metric-chip">相似度 {src['score']:.3f}</span>
                    <br><small>{src['text'][:200]}{"..." if len(src['text']) > 200 else ""}</small>
                    </div>
                    """, unsafe_allow_html=True)


# ====== 处理新问题 ======
def process_question(user_query: str):
    """处理用户问题，执行 RAG 检索+生成"""
    from rag_engine import rag_answer

    with st.spinner("🔍 正在检索知识库并生成回答..."):
        try:
            from rag_engine import rag_answer, post_process_answer

            answer, retrieved = rag_answer(
                query=user_query,
                api_key=st.session_state.api_key,
                base_url=st.session_state.base_url,
                model_name=st.session_state.model_name,
                top_k=st.session_state.top_k,
                stream=False
            )
            # 如果 answer 是生成器，收集完整内容
            if hasattr(answer, '__iter__') and not isinstance(answer, str):
                answer = "".join(answer)

            # 后处理：修正小模型数字幻觉（如2223→2023）
            answer = post_process_answer(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": retrieved
            })
        except Exception as e:
            err_msg = f"❌ 生成回答时出错：{str(e)}\n\n请检查 API Key 配置，或切换到演示模式（不填 API Key）。"
            st.session_state.messages.append({
                "role": "assistant",
                "content": err_msg,
                "sources": []
            })


# 如果最后一条消息是用户消息且没有对应回答，立即处理
if (st.session_state.messages and
        st.session_state.messages[-1]["role"] == "user"):
    user_q = st.session_state.messages[-1]["content"]
    process_question(user_q)
    st.rerun()


# ====== 输入框 ======
st.markdown("---")
with st.form("chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "输入您的金融问题",
            placeholder="例如：比亚迪2023年上半年毛利率是多少？",
            label_visibility="collapsed"
        )
    with col_btn:
        submitted = st.form_submit_button("发送 ➤", use_container_width=True, type="primary")

    if submitted and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        st.rerun()


# ====== 底部说明 ======
with st.expander("📖 系统说明 & 技术架构"):
    st.markdown("""
    ### 技术架构

    ```
    用户问题
        ↓
    Embedding 向量化（multilingual-MiniLM）
        ↓
    FAISS 向量数据库检索（Top-K 相关片段）
        ↓
    Prompt 构建（角色设定 + 参考资料 + 问题）
        ↓
    LLM 生成回答（DeepSeek / OpenAI 兼容 API）
        ↓
    结构化输出（回答 + 引用来源）
    ```

    ### 知识库覆盖（30份文档）
    | 类别 | 文档 |
    |------|------|
    | 企业年报/半年报 | 比亚迪、宁德时代、贵州茅台、中国平安、阿里巴巴、腾讯、招商银行、美的、格力、隆基绿能、中国中免、海天味业、迈瑞医疗、万科、工商银行、中国石油、五粮液、理想汽车、拼多多 |
    | 行业研究报告 | 新能源汽车、白酒、银行、互联网、半导体、光伏、医药、AI大模型 |
    | 宏观经济报告 | 2023年A股市场与宏观经济回顾 |

    ### 使用说明
    - **演示模式**（不填 API Key）：直接展示知识库检索到的原文，适合验证检索效果
    - **LLM模式**（填入 API Key）：基于检索内容调用大模型生成专业分析回答
    - 支持跨文档横向对比分析（如"比较茅台和五粮液的毛利率"）
    """)

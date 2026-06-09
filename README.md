# 金融投研 RAG 智能问答系统

基于 **RAG（检索增强生成）** 技术的金融投研问答系统，覆盖 30 份 A 股权威财报与行业研报。

## 功能特性

- 📚 **30 份知识库文档**：比亚迪、宁德时代、贵州茅台、腾讯、阿里巴巴等头部企业年报 + 行业研报
- 🔍 **FAISS 向量检索**：多语言 MiniLM 嵌入，余弦相似度 Top-K 检索
- 🤖 **LLM 接入**：支持 DeepSeek / SiliconFlow / OpenAI 兼容接口
- 💬 **聊天界面**：Streamlit 多轮对话，展示参考来源

## 在线使用

直接访问 Streamlit Cloud 部署版本，无需安装。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 技术架构

```
用户提问 → SentenceTransformer 嵌入 → FAISS 检索 Top-K
→ Prompt 构建 → LLM 生成回答 → 展示来源
```

> ⚠️ 本系统仅供学习研究，不构成任何投资建议。

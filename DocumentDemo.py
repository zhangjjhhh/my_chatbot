from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ==================== 配置大模型（虽然今天不用，但先配上）====================
llm = ChatOpenAI(
    api_key="sk-wsthjcucuojcymksgvdovsrfdzxtvoybddrcjpcrkigltsqf",
    base_url="https://api.siliconflow.cn/v1",
    model="deepseek-ai/DeepSeek-V3",
    temperature=0.1
)

# ==================== 1. 加载本地 TXT 文件 ====================
# 文件路径：当前文件夹下的 info.txt
loader = TextLoader("docs/studyPlan.txt", encoding="utf-8")
# 执行加载，得到文档列表
documents = loader.load()

# 2. 【今天核心】文本切分器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,       # 每块最大长度
    chunk_overlap=20,      # 重叠长度（保证语义连贯）
    length_function=len,   # 按字符长度计算
)
# 3. 执行切分
splits = text_splitter.split_documents(documents)

# =====================  今天核心：配置 Embedding 模型 =====================
# 硅基流动 向量模型
embeddings = OpenAIEmbeddings(
    api_key="sk-wsthjcucuojcymksgvdovsrfdzxtvoybddrcjpcrkigltsqf",
    base_url="https://api.siliconflow.cn/v1",
    model="BAAI/bge-m3"  # 向量模型专用
)

# ===================== 3. 【今天核心】存入 Chroma 向量库 =====================
db = Chroma.from_documents(
    documents=splits,     # 切片后的文档
    embedding=embeddings  # 向量模型
)
# ===================== 4. 创建检索器 retriever =====================
retriever = db.as_retriever(
    search_kwargs={"k": 2}  # 只返回最相似的2条
)

# ===================== 5. 测试检索 =====================
query = "45天学习的是什么内容？"
retrieved_docs = retriever.invoke(query)

print("🔍 检索到的相关内容：")
print("-" * 50)
for i, doc in enumerate(retrieved_docs):
    print(f"结果 {i+1}:")
    print(doc.page_content)
    print("-" * 50)

# ===================== 3. 测试：把一句话转向量 =====================
test_text = "我正在学习 RAG 开发"
vector = embeddings.embed_query(test_text)

print("✅ 测试文本：", test_text)
print("✅ 向量长度（维度）：", len(vector))
print("✅ 前10个向量数字：", vector[:10])

# ===================== 4. 把所有文档片段转向量 =====================
print("\n===== 开始批量生成文档向量 =====")
doc_vectors = embeddings.embed_documents([chunk.page_content for chunk in splits])

print(f"✅ 共生成 {len(doc_vectors)} 个文档向量")
print(f"✅ 每个向量维度：{len(doc_vectors[0])}")

# 4. 查看结果
print(f"切分后总块数：{len(splits)}")
print("="*50)

for i, chunk in enumerate(splits):
    print(f"第 {i+1} 块：")
    print(chunk.page_content)
    print("-"*50)

# ==================== 2. 查看加载结果 ====================
print("✅ 成功加载文档！")
print(f"文档数量：{len(documents)}")
print("\n📄 文档内容：")
print(documents[0].page_content)

# 元数据（来源、日期等）
print("\n📌 元数据：")
print(documents[0].metadata)
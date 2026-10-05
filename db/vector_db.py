# db/vector_db.py
import chromadb
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIR, COLLECTION_MARKET, COLLECTION_COURSE, EMBEDDING_MODEL
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from utils.logging_config import setup_logging

logger = setup_logging()

_chroma_client = None
_market_collection = None
_course_collection = None

# ------------ 初始化 Chroma 向量数据库
def init_vectorstores():


    global _chroma_client, _market_collection, _course_collection
    settings = Settings(
        # 指定 Chroma 数据持久化目录（config.CHROMA_PERSIST_DIR，当前为 data/chroma_db_v2/），
        # 所有向量和元数据将保存于此。
        persist_directory = CHROMA_PERSIST_DIR,
        # 禁用 Chroma 的匿名使用统计上报，保护隐私。
        anonymized_telemetry = False
    )
    # 创建 Chroma 客户端实例，该客户端以嵌入式模式运行，无需外部服务。
    _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=settings)

    # 创建中文嵌入函数：整个项目只建这一个实例，存文档和查文档都用它，
    # 保证同一句话在"入库"和"检索"两端被编码成同一个向量。
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    _market_collection = _chroma_client.get_or_create_collection(
        # 根据 name 获取已有集合，若不存在则新建。
        name=COLLECTION_MARKET,
        # 指定使用余弦相似度作为向量检索的距离度量，适用于语义相似度比较。
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_fn 
    )
    _course_collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_COURSE,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_fn
    )
    logger.info("向量数据库初始化完成，集合：%s, %s", COLLECTION_MARKET, COLLECTION_COURSE)
    return _chroma_client, _market_collection, _course_collection


# --检查全局变量是否为 None，若是则调用 init_vectorstores() 完成初始化，然后返回集合对象。
def get_market_collection():
    if _market_collection is None:
        init_vectorstores()
    return _market_collection

def get_course_collection():
    if _course_collection is None:
        init_vectorstores()
    return _course_collection
# ----------------------------------------------#

# ----------向向量集合添加文档
def add_documents_to_collection(collection_name: str, documents: list, metadatas: list = None, ids: list = None):
    #  根据名称获取集合
    if collection_name == COLLECTION_MARKET:
        coll = get_market_collection()
    elif collection_name == COLLECTION_COURSE:
        coll = get_course_collection()
    else:
        raise ValueError(f"collection_name 必须是 '{COLLECTION_MARKET}' 或 '{COLLECTION_COURSE}'")
    
    # 处理元数据和 ID 默认值
    if metadatas is None:
        metadatas = [{}] * len(documents)
    if ids is None:
        import uuid
        ids = [str(uuid.uuid4()) for _ in documents]
    if len(documents) != len(metadatas) or len(documents) != len(ids):
        raise ValueError("documents, metadatas, ids 长度必须相同")
    
    # ===== 新增：分批添加，避免超限 =====
    BATCH_SIZE = 100  # 安全值，低于 ChromaDB 限制 166
    total = len(documents)
    for i in range(0, total, BATCH_SIZE):
        end = min(i + BATCH_SIZE, total)
        batch_docs = documents[i:end]
        batch_metas = metadatas[i:end]
        batch_ids = ids[i:end]
        coll.add(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
        logger.info("已提交批次 %d-%d，共 %d 条", i+1, end, end-i)
    logger.info("向量数据库更新成功，集合 %s，新增 %d 条", collection_name, total)

def query_collection(collection_name: str, query_text: str, n_results: int = 3):
    if collection_name == COLLECTION_MARKET:
        coll = get_market_collection()
    elif collection_name == COLLECTION_COURSE:
        coll = get_course_collection()
    else:
        raise ValueError(f"collection_name 必须是 '{COLLECTION_MARKET}' 或 '{COLLECTION_COURSE}'")
    
    results = coll.query(query_texts=[query_text], n_results=n_results, include=["documents", "metadatas"])
    docs = results['documents'][0] if results['documents'] else []
    metadatas = results['metadatas'][0] if results['metadatas'] else []
    return docs, metadatas
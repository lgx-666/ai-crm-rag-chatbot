# import_knowledge.py
import os
import sys
from file_loader import load_and_chunk_file
from db.vector_db import add_documents_to_collection
import hashlib
from config import COLLECTION_MARKET, COLLECTION_COURSE

def import_files(directory: str, collection: str, extensions=('.pdf', '.docx', '.md', '.html', '.txt')):
    """
    递归遍历目录，导入所有指定扩展名的文件到指定集合
    """
    if collection not in (COLLECTION_MARKET, COLLECTION_COURSE):
        print(f"collection 必须为 '{COLLECTION_MARKET}' 或 '{COLLECTION_COURSE}'")
        return

    all_chunks = []
    metadatas = []
    # 递归遍历目录（os.walk 循环）
    #   os.walk(directory)：生成一个三元组迭代器，每次迭代返回 (root, dirs, files)
    #   root：当前正在遍历的目录完整路径（例如 "./knowledge/llm"）。
    #   dirs：当前目录下的子目录名称列表（例如 ['papers']）。
    #   files：当前目录下的文件名称列表（例如 ['transformer.pdf', 'notes.txt']）。
    for root, dirs, files in os.walk(directory):
        # 内部循环：遍历当前目录下的每个文件
        for file in files:
            # 文件扩展名过滤，检查当前文件名是否以允许的扩展名结尾（忽略大小写）。
            if file.lower().endswith(extensions):
                #  构建完整文件路径：将当前目录 root 和文件名 file 拼接成完整的文件路径。
                file_path = os.path.join(root, file)
                print(f"正在处理: {file_path}")
                # 尝试解析文件并分块
                try:
                    # 根据文件后缀自动选择解析器，将文本切分为多个块，每块最大 500 字符，块之间重叠 50 字符（保证连贯性）。
                    chunks = load_and_chunk_file(file_path, chunk_size=500, overlap=50)
                    # 返回：一个字符串列表 chunks，例如 ["文本块1...", "文本块2...", ...]。
                    for chunk in chunks:
                        all_chunks.append(chunk)
                        metadatas.append({"source": file, "path": file_path})
                except Exception as e:
                    import traceback
                    print(f"处理失败: {e}")
                    traceback.print_exc()   # 打印详细堆栈

    if not all_chunks:
        print("未提取到任何文本")
        return

        # 生成 ID 并批量入库
    # 为什么不再用 uuid.uuid4()：
    #   uuid4 每次运行都是全新随机值，Chroma 认不出"这段文本上次已经导过"，
    #   于是重复执行本脚本会把同一批内容再灌一遍，块数成倍增长
    #   （旧库 6197 块 对比 新切块 672 块，差额就是这么累积出来的）。
    #   改成"文本内容的 md5 哈希"作主键后：同一段文本永远得到同一个 ID，
    #   重复导入等于覆盖（幂等），同一次导入内部的重复段落也会被顺带去重。
    # 注意：这里同时压缩 all_chunks / metadatas / ids 三个列表，
    #       保证三者长度始终一致——vector_db.add_documents_to_collection 会做长度校验。
    seen, unique_chunks, unique_metas, ids = set(), [], [], []
    for chunk, meta in zip(all_chunks, metadatas):
        cid = hashlib.md5(chunk.encode("utf-8")).hexdigest()
        if cid in seen:
            continue                 # 同内容已收过，跳过
        seen.add(cid)
        unique_chunks.append(chunk)
        unique_metas.append(meta)
        ids.append(cid)

    add_documents_to_collection(collection, unique_chunks,
                                metadatas=unique_metas, ids=ids)
    print(f"成功导入 {len(unique_chunks)} 个文档块到集合 {collection}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("用法: python import_knowledge.py <目录路径> <market|java>")
        sys.exit(1)
    dir_path = sys.argv[1]
    col = sys.argv[2]
    import_files(dir_path, col)


# python import_knowledge.py <目录路径> <集合名称>
# 目录路径:  存放知识文件（PDF/Word/Markdown等）的文件夹路径。可以是绝对路径或相对路径
# 集合名称:  指定将知识存入哪个 Chroma 集合。
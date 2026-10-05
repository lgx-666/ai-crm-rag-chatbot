# file_loader.py
import os
import re
import pypdf
from typing import List
from docx import Document
import markdown
from bs4 import BeautifulSoup
import chardet
from utils.logging_config import setup_logging
logger = setup_logging()  

# PDF处理器
# def extract_text_from_pdf(file_path: str) -> str:
#     """从 PDF 提取文本"""

#     text = ""
#     try:
#         # pdfplumber.open(file_path)：以上下文管理器打开 PDF 文件。
#         with pdfplumber.open(file_path) as pdf:
#             # for page in pdf.pages：遍历每一页。
#             for page in pdf.pages:
#                 # page.extract_text()：提取当前页的文本内容（如果该页是扫描图片则返回 None）。
#                 page_text = page.extract_text()
#                 # 拼接：每页文本追加换行符，保留页面边界。
#                 if page_text:
#                     text += page_text + "\n"
#         # 注意：对于纯图片 PDF（扫描件），extract_text 可能返回空字符串，
#         # 此时需要 OCR 支持（如 pytesseract），该处并未实现，后续可能会处理。
#     except Exception as e:
#         logger.error(f"PDF 提取失败: {file_path}, 错误: {e}")
#         # 可以重新抛出或返回空字符串，这里选择返回空
#     return text


def extract_text_from_pdf(file_path: str) -> str:
    """从 PDF 提取文本（使用 pypdf 加速）"""
    text = ""
    try:
        reader = pypdf.PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        logger.error(f"PDF 提取失败: {file_path}, 错误: {e}")
    return text


# world处理器
def extract_text_from_docx(file_path: str) -> str:
    """从 Word 文档提取文本"""

    # Document(file_path)：加载 Word 文档（python-docx 库）。
    doc = Document(file_path)

    # doc.paragraphs：遍历所有段落。
    # para.text：提取段落的纯文本。
    # 拼接：用换行符连接所有段落。
    return "\n".join([para.text for para in doc.paragraphs])

# markdown处理器
def extract_text_from_markdown(file_path: str) -> str:
    """从 Markdown 文件提取文本（去除标记）"""

    # 读入 MD 文件：以 UTF-8 编码读取原始文本。
    with open(file_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
    
    # markdown.markdown(md_text)：将 Markdown 语法转换为 HTML 字符串（例如 # 标题 → <h1>标题</h1>）。
    html = markdown.markdown(md_text)
    # BeautifulSoup(html, 'html.parser')：解析 HTML 文档树。
    soup = BeautifulSoup(html, 'html.parser')
    # soup.get_text()：提取所有可见文本，去除 HTML 标签（如 <h1>、<p> 等），只保留纯文本内容。
    return soup.get_text()

# HTML网页处理器
def extract_text_from_html(file_path: str) -> str:
    """从 HTML 文件提取文本"""

    with open(file_path, 'r', encoding='utf-8') as f:
        html_text = f.read()
    soup = BeautifulSoup(html_text, 'html.parser')
    return soup.get_text()

# txt文本处理器
def extract_text_from_txt(file_path: str) -> str:
    """从纯文本文件提取文本（自动检测编码）"""

    # 二进制读取：以 'rb' 模式读取，避免编码错误。
    with open(file_path, 'rb') as f:
        raw = f.read()
        # chardet.detect(raw)：自动检测文件编码（如 UTF-8、GBK、GB2312 等）。
        encoding = chardet.detect(raw)['encoding'] or 'utf-8'   #回退编码：如果检测失败，默认使用 'utf-8'。
    # 二次读取：用检测到的编码重新读取文件，返回正确解码的文本。
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()

def extract_text(file_path: str) -> str:
    """根据扩展名调用相应的提取函数"""

    # os.path.splitext(file_path)：
    #   这是 Python os.path 模块提供的函数，用于将路径拆分为 "主体部分" 和 "扩展名部分"。
    #   它返回一个元组 (root, ext)，其中：
    #   root 是去除扩展名后的部分（包含目录路径和主文件名）。
    #   ext 是扩展名（包含点号，如 .pdf）。
    # [1]：
    #   提取元组中的第二个元素，即扩展名字符串（含点号）。
    # .lower()：
    #   将扩展名字符串中的所有字母转换为小写，实现大小写不敏感的匹配。
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext == '.md':
        return extract_text_from_markdown(file_path)
    elif ext == '.html' or ext == '.htm':
        return extract_text_from_html(file_path)
    elif ext == '.txt':
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50,
            min_chunk: int = 30) -> List[str]:
    """
    将长文本切分成多个小块（chunk）
    chunk_size: 每块最大字符数
    overlap: 块之间的重叠字符数（保持上下文连贯）
    min_chunk: 短于该长度的块直接丢弃（页眉、"扫码加查看更多"这类无信息行）
    """

    # 短文本无需切分；但仍要过一遍最小长度门槛，
    # 否则几个字的文本也会入库，变成一个能抢走召回位置的检索噪声。
    if len(text) <= chunk_size:
                # 必须和下面聚合分支的产出表示一致：逐行去空白后用单个空格连接，
        # 否则同一库里短块带 \n、长块不带 \n，向量化结果和主键都对不上。
        stripped = " ".join(ln.strip() for ln in text.split("\n") if ln.strip())
        return [stripped] if len(stripped) >= min_chunk else []

    # 逐行清洗：去首尾空白、丢掉空行。空行只当段落分隔用，本身不进块。
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]

    chunks = []     # 存最终切出的块
    buf = []        # 当前正在累加的行
    buf_len = 0     # 缓冲区已累计的字符数
    # 固定步长：每块最多前进 chunk_size - overlap 个字符。
    # 原实现是 start = max(end - overlap, start + 1)：当断句点 end 紧贴 start 时，
    # end - overlap 会小于 start + 1，步长退化成 1 →
    # 同一句话被"每次少一个字符"地重复几十遍，产出大量 1~40 字的近重复块
    # （实测占全库 66%，6197 块里约 4110 块是这种垃圾）。
    # 现在步长与标点位置彻底解耦。
    stride = max(1, chunk_size - overlap)

    for line in lines:
        line_len = len(line)

        # 超长单行（PDF 里整段不换行很常见）：先把缓冲区落成一块，
        # 再对这一行本身做定长滑窗，保证它不会因为一行太长而整块丢弃。
        if line_len > chunk_size:
            if buf:
                chunks.append(" ".join(buf).strip())
                buf, buf_len = [], 0
            for i in range(0, line_len, stride):
                piece = line[i:i + chunk_size]
                if len(piece.strip()) >= min_chunk:
                    chunks.append(piece.strip())
            continue

        buf.append(line)
        buf_len += line_len + 1      # +1 是给行间拼接时插入的那个空格
        if buf_len >= chunk_size:    # 攒够一块就落盘
            chunks.append(" ".join(buf).strip())
            buf, buf_len = [], 0

    if buf:                          # 收尾：最后不满一块的剩余内容
        chunks.append(" ".join(buf).strip())

    # 最后一道过滤：丢掉过短碎片，避免它们以"高相似短句"的身份抢走正文的召回位。
    return [c for c in chunks if len(c) >= min_chunk]

def load_and_chunk_file(file_path: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    full_text = extract_text(file_path)
    print(f"提取文本长度: {len(full_text)}")   # 临时调试
    if not full_text or not full_text.strip():
        return []   # 无有效文本，直接返回空列表
        # 原来这里是 re.sub(r'\s+', ' ', ...)：\s 包含 \n，会把整篇压成一个没有任何断点的长字符串。
    # 而换行恰恰是 pypdf 唯一给出的版面信号（每页末尾追加了一个 \n），
    # 压掉它，上面的按行聚合就退化成"整篇一行"，只能走滑窗，段落边界全丢。
    # 现在只压"水平方向"的空白（空格/制表/回车/换页），保留 \n，并把连续空行并成一个换行。
    full_text = re.sub(r'[ \t\r\f\v]+', ' ', full_text)
    full_text = re.sub(r'\n\s*\n+', '\n', full_text).strip()
    return chunk_text(full_text, chunk_size, overlap)
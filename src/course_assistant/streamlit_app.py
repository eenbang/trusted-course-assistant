import sys
import hashlib
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

# 导入之前已经写好的“读取 PDF、检索片段、生成回答”功能。
from answer import answer_with_sources
from retrieval import load_pdf, retrieve


# 设置网页的标题和浏览器标签页文字。
st.set_page_config(page_title="可信课程资料研究助手")
st.title("可信课程资料研究助手")
st.write("上传课程 PDF，提出问题，并查看回答所依据的资料页码。")

# 让用户选择一个 PDF 文件。
uploaded_file = st.file_uploader(
    "上传课程 PDF",
    type=["pdf"],
)

# 让用户输入问题。
question = st.text_input(
    "你想问什么？",
    placeholder="例如：什么是深度学习？",
)

# 点击按钮后才开始读取 PDF、检索资料和调用模型。
if st.button("提问"):
    if uploaded_file is None:
        st.warning("请先上传一个 PDF 文件。")

    elif not question.strip():
        st.warning("请先输入问题。")

    else:
        # 显示处理中提示，避免等待时页面看起来没有反应。
        with st.spinner("正在读取资料并生成回答……"):
            # Streamlit 每次运行脚本时都会重新执行本段代码。
            # 用上传内容的哈希生成稳定文件名，让同一份 PDF 复用本地缓存。
            uploaded_bytes = uploaded_file.getvalue()
            file_hash = hashlib.sha256(uploaded_bytes).hexdigest()

            # data/documents 已在 .gitignore 中忽略，不会提交到 GitHub。
            documents_dir = PROJECT_ROOT / "data" / "documents"
            documents_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = documents_dir / f"{file_hash}.pdf"

            # 只有第一次上传这份文件时才写入本地；后续直接复用。
            if not pdf_path.exists():
                pdf_path.write_bytes(uploaded_bytes)

            try:
                # 读取 PDF，并切成可用于检索的文字片段。
                chunks = load_pdf(pdf_path)

                # load_pdf 使用临时文件名作为来源名称。
                # 这里改成用户上传时看到的原始文件名，便于识别来源。
                for chunk in chunks:
                    chunk.source = uploaded_file.name

                if not chunks:
                    st.error(
                        "没有从 PDF 中提取到文字。"
                        "它可能是扫描版图片，暂时需要 OCR 才能读取。"
                    )
                else:
                    # 找出与问题较相关的文字片段。
                    results = retrieve(question, chunks)

                    if not results:
                        st.warning("没有检索到相关片段，暂时无法依据这份资料回答。")
                    else:
                        # 根据检索片段生成带资料编号的回答。
                        answer = answer_with_sources(question, results)

                        st.subheader("回答")
                        st.markdown(answer)

                        # 展开后可以检查模型回答依据的原文。
                        with st.expander("查看检索到的资料原文"):
                            for number, (chunk, score) in enumerate(
                                results,
                                start=1,
                            ):
                                st.markdown(
                                    f"**[资料{number}] "
                                    f"{chunk.source}，PDF 第 {chunk.page} 页**"
                                )
                                st.caption(f"片段匹配分数：{score:.3f}")
                                st.write(chunk.text)

            except Exception as error:
                # 显示错误信息，方便发现 API 密钥、网络或 PDF 读取问题。
                st.error(f"处理失败：{error}")

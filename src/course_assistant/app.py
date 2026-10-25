import argparse
from pathlib import Path

from answer import answer_with_sources
from retrieval import load_pdf, retrieve


def main():
    # 定义命令行输入：PDF 路径和用户问题。
    parser = argparse.ArgumentParser(
        description="根据课程 PDF 检索资料并生成回答"
    )
    parser.add_argument("pdf", type=Path, help="PDF 文件路径")
    parser.add_argument("question", help="要询问的问题")
    args = parser.parse_args()

    if not args.pdf.is_file():
        parser.error(f"找不到 PDF 文件：{args.pdf}")

    # 第一步：读取 PDF，提取文字并切成片段。
    chunks = load_pdf(args.pdf)

    # 第二步：从片段中找与问题相关的内容。
    results = retrieve(args.question, chunks)

    if not results:
        print("没有检索到片段，暂时无法依据这份资料回答。")
        return

    # 第三步：把检索结果交给模型生成答案。
    answer = answer_with_sources(args.question, results)
    print("\n回答：")
    print(answer)

    # 单独打印编号和来源，方便检查回答中的引用。
    print("\n检索来源：")
    for number, (chunk, score) in enumerate(results, start=1):
        print(
            f"[资料{number}] {chunk.source}，"
            f"PDF 第 {chunk.page} 页（匹配分数 {score:.3f}）"
        )


if __name__ == "__main__":
    main()
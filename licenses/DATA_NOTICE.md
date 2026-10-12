# 数据来源与使用说明

作品：Lewis Carroll, Alice's Adventures in Wonderland。电子书提供者：Project Gutenberg，编号11；来源页 https://www.gutenberg.org/ebooks/11；下载链接 https://www.gutenberg.org/ebooks/11.txt.utf-8。

来源页标示 Public domain in the USA.。这不是对所有司法辖区的版权结论，也不能把 nanoGPT 的 MIT 许可证当作本数据的许可证。

运行 prepare_data.py 后生成原始完整文本：data/alice_original.txt，包含电子书提供者的完整使用说明及许可条款。处理版本：data/input.txt，仅取 START/END 标记之间的正文，CRLF 转 LF、裁掉外围空白、尾部补换行；其中仍保留正文前的目录等内容。具体时间和 SHA256 见 data/source.json。

公开Git仓库不包含语料全文，包含来源元数据与下载脚本；原文和处理版只保存在本地并被Git忽略。本项目在本地非收费教学实验中使用。若复制或再发布原始电子书或此分发包，应阅读随附的 Gutenberg 条款并遵守适用要求。项目代码的 MIT 许可不覆盖第三方电子书及第三方包。

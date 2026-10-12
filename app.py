"""仅绑定本机的教学展示界面；不需要 API key。"""
import argparse
import csv
import os
from pathlib import Path
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr
import pandas as pd
import torch
from generate import load_checkpoint, generate_text
from runtime import ROOT


def create_handler(model, tokenizer):
    def predict(prompt, length, temperature, top_k, seed):
        """将 UI 输入交给与命令行相同的生成函数，避免两套采样逻辑。"""
        try:
            unknown = sorted(set(prompt) - set(tokenizer.stoi))
            if unknown:
                gr.Warning("输入含词表外字符，模型会按 UNK 理解：" + "".join(unknown[:12]))
            return generate_text(model, tokenizer, prompt, int(length), float(temperature), int(top_k) or None, int(seed))
        except (ValueError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc
    return predict


def build_demo(checkpoint, device="auto"):
    checkpoint = Path(checkpoint)
    model, tokenizer, info = load_checkpoint(checkpoint, device)
    points = []
    metrics = checkpoint.parent / "metrics.csv"
    if metrics.is_file():
        with metrics.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                for split in ("train", "val"):
                    if row.get(split + "_loss"):
                        points.append({"step": int(row["step"]), "loss": float(row[split + "_loss"]), "split": split})
    with gr.Blocks(title="MiniGPT 字符级 Transformer 实验室") as demo:
        gr.Markdown("# MiniGPT 字符级 Transformer 实验室\n从零训练的小模型：观察字符如何逐个生成。短训练输出可能不连贯。")
        gr.Markdown(f"**参数量** {model.parameter_count():,} · **注意力头** {model.config.n_head} · **层数** {model.config.n_layer} · **上下文** {model.config.block_size} 字符 · **设备** {next(model.parameters()).device}\n\n当前权重：`{checkpoint.parent.name}/{checkpoint.name}`，保存于第 {info['step']} 步，验证 Loss = {info['losses']['val']:.4f}。")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="起始文本（训练语料为英文）", value="Alice", lines=3)
                length = gr.Slider(1, 500, value=200, step=1, label="新增字符数")
                temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature：越低越保守")
                top_k = gr.Slider(0, len(tokenizer.chars), value=0, step=1, label="Top-k：0 表示不限制")
                seed = gr.Number(value=42, precision=0, minimum=0, maximum=2**32-1, label="采样随机种子")
                button = gr.Button("生成文本", variant="primary")
            with gr.Column():
                output = gr.Textbox(label="生成结果（包含起始文本）", lines=15)
        button.click(create_handler(model, tokenizer), [prompt, length, temperature, top_k, seed], output, api_name="generate", concurrency_limit=1)
        if points:
            gr.LinePlot(pd.DataFrame(points), x="step", y="loss", color="split", title="固定窗口评估 Loss（nats/字符）", x_title="优化器更新次数", y_title="交叉熵")
        with gr.Accordion("模型结构与实验说明", open=False):
            gr.Markdown("字符编号 → Token Embedding + Position Embedding → 2 条残差支路的 Transformer Blocks → LayerNorm → 字符 logits。\n\n每个位置仅能关注自身和前文。单头与四头保持总宽度和参数量相等，但每头维度、注意力矩阵数量和运行开销不同。完整实验见 runs/comparison/comparison.json。\n\n曲线来自该运行的全部评估点；当前加载的 best.pt 可能早于曲线终点。验证集用于选择 checkpoint，未设置独立测试集。")
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/default/best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    torch.set_num_threads(2)
    build_demo(args.checkpoint, args.device).queue(max_size=8).launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False)

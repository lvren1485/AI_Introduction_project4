# 大语言模型部署与对比分析项目

## 🌟 项目简介 (Project Overview)

本项目是985大学软件工程专业《人工智能导论》课程的第四次作业成果。本项目的核心目标是**亲自动手部署并测试主流的大语言模型**，从而深入理解其工作原理、部署流程，并对其在实际问答场景中的表现进行横向对比分析。通过本次实践，我们旨在将课堂上学习的理论知识与实际操作相结合，直观感受不同大语言模型在语义理解、逻辑推理和语言表达方面的特点与差异。

本项目主要部署并对比了以下两款国产开源大语言模型：

- **Qwen-7B-Chat (通义千问7B对话版)**
- **ChatGLM3-6B (智谱清言3代6B)**

## 🚀 部署环境 (Deployment Environment)

本项目的部署和测试均在**魔搭 (ModelScope) 平台**上完成。魔搭平台为我们提供了便捷的云计算资源（CPU服务器）和预配置的Jupyter Notebook开发环境，极大地简化了大型模型部署的复杂性。

### 环境配置与依赖

为了确保模型能够顺利运行，我们进行了以下环境配置：

1. **注册并登录ModelScope**：绑定阿里云账号以获取免费CPU计算资源。

2. **启动CPU服务器**：在ModelScope的"我的Notebook"界面启动CPU环境。

3. **进入终端环境**：通过Jupyter Notebook界面内的Terminal，打开命令行终端。

4. **环境搭建（可选Miniconda）**：根据教程指引，若CPU环境未预装Conda，则手动下载并安装Miniconda，创建并激活`qwen_env`环境。

5. **安装基础依赖**：

   ```
   pip install torch==2.3.0+cpu torchvision==0.18.0+cpu --index-url https://download.pytorch.org/whl/cpu
   pip install -U pip setuptools wheel
   pip install "intel-extension-for-transformers==1.4.2" "neural-compressor==2.5" "transformers==4.33.3" "modelscope==1.9.5" "pydantic==1.10.13" "sentencepiece" "tiktoken" "einops" "transformers_stream_generator" "uvicorn" "fastapi" "yacs" "setuptools_scm"
   pip install fschat --use-pep517
   pip install tqdm huggingface-hub # 可选增强体验
   ```

### 模型下载

在`/mnt/data`目录下，使用`git clone`命令下载所需的大语言模型。**注意：为避免存储不足，建议一次只下载并测试一个模型。**

```
cd /mnt/data
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
git clone https://www.modelscope.cn/qwen/Qwen-7B-Chat.git
```

**部署完成截图示例：** （此处应插入`git clone`命令执行成功的终端截图，或ModelScope平台中模型部署完成的截图，展示模型文件已成功下载到`/mnt/data`目录或部署状态为“运行中”。）

## 💻 如何运行 (How to Run)

在模型文件下载完成后，您可以通过编写Python推理脚本来加载模型并进行问答测试。

### 推理脚本示例 (`run_inference.py`)

```
from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
import os

# 根据你想要测试的模型修改路径
# model_name_qwen = "/mnt/data/Qwen-7B-Chat"
model_name_chatglm = "/mnt/data/chatglm3-6b"

# 选择当前要加载的模型路径
current_model_path = model_name_chatglm # 或 model_name_qwen

# 定义测试问题列表
prompts = [
    "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少",
    "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上。",
    "他知道我知道你知道他不知道吗？ 这句话里，到底谁不知道？",
    "明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？",
    "领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。请问：以上“意思”分别是什么意思。"
]

def run_model_inference(model_path, prompts_list):
    """
    加载指定模型并对给定的问题列表进行推理。
    """
    print(f"\n--- 正在加载模型: {os.path.basename(model_path)} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto"  # 自动选择 float32/float16（根据模型配置）
        ).eval()

        streamer = TextStreamer(tokenizer)

        for i, prompt_text in enumerate(prompts_list):
            print(f"\n====== 问题 {i+1}/{len(prompts_list)} ======")
            print(f"提问: {prompt_text}")

            # 对于ChatGLM模型，可能需要特定的对话格式
            if "chatglm" in model_path.lower():
                history = []
                messages = [{"role": "user", "content": prompt_text}]
                inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = tokenizer(inputs, return_tensors="pt").to(model.device)
                
                print("模型回答:")
                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=0.8, temperature=0.7, repetition_penalty=1.0)
                response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
                print(response)
            else: # Qwen模型或其他通用模型
                inputs = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
                print("模型回答:")
                # 这里为了简化，直接使用streamer，但在实际保存时可能需要捕获输出
                outputs = model.generate(inputs, streamer=streamer, max_new_tokens=512)


    except Exception as e:
        print(f"加载或运行模型 {os.path.basename(model_path)} 时发生错误: {e}")

# 运行Qwen-7B-Chat
# run_model_inference("/mnt/data/Qwen-7B-Chat", prompts)

# 运行ChatGLM3-6B
run_model_inference("/mnt/data/chatglm3-6b", prompts)
```

### 运行指令

在终端中执行Python脚本：

```
cd /mnt/workspace # 确保你在工作目录下
python run_inference.py
```

## 📊 模型横向对比分析 (Model Comparison Analysis)

本项目对Qwen-7B-Chat和ChatGLM3-6B两款模型在以下五个特定问答场景中的表现进行了详细对比。完整的对比分析报告和测试结果截图，请参考项目根目录下的 [大语言模型部署与对比分析报告.md](大语言模型部署与对比分析报告.md) （或您报告的实际文件名）。

以下是我们在测试中使用的核心问题列表，旨在全面考察模型的语义理解、逻辑推理、语境辨析和中文表达能力：

1. **语义歧义辨析**：`请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少`
2. **隐含语义与社会洞察**：`请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上。`
3. **复杂逻辑推理与悖论处理**：`他知道我知道你知道他不知道吗？ 这句话里，到底谁不知道？`
4. **中文指代与重叠词理解**：`明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？`
5. **多义词语境辨析**：`领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。请问：以上“意思”分别是什么意思。`

### 对比亮点概述

通过对上述问题的测试，我们观察到两款模型各有侧重：

- **Qwen-7B-Chat**：
  - **语义理解深度**：在处理复杂语境、隐含意图和带有情感色彩的问题时，展现出更强的深层语义理解能力，回答更富洞察力。
  - **语言表达自然度**：其生成的回应更接近人类的口语习惯，流畅自然，甚至能体现一定的幽默感和共情能力。
  - **推理灵活性**：在面对开放性或需要联想的问题时，表现出更灵活的推理路径，能给出更具启发性的答案。
- **ChatGLM3-6B**：
  - **回答简洁精准**：在获取核心信息和给出直接答案方面表现出色，回答风格直接、干练。
  - **逻辑严谨性**：在处理逻辑性问题时，更偏向于形式逻辑分析，回答严谨、不易产生歧义。
  - **字面意义把握**：对词语的字面含义和固定搭配的理解准确到位。

**总体而言**，Qwen-7B-Chat在处理需要深度理解、灵活推理和富有情感色彩的中文交互中表现更优；而ChatGLM3-6B则在提供简洁、精确、结构化信息方面更具优势。这表明在实际应用中，应根据具体的任务需求来选择最适合的大语言模型。

## 📝 完整报告 (Full Report)

您可以在本项目的根目录下找到详细的对比分析报告，其中包含了部署截图、完整的问答测试结果截图以及更深入的对比分析：

- [大语言模型部署与对比分析报告.md](大语言模型部署与对比分析报告.md) (请根据您的实际文件名修改此链接)

## ✨ 未来展望 (Future Work)

本次作业只是大语言模型探索的起点。未来，我们计划：

1. **尝试更大规模模型**：在条件允许的情况下，部署和测试更大参数量的大语言模型，观察其性能提升。
2. **研究模型量化技术**：探索如何对模型进行量化，以降低其部署所需的计算资源和存储空间，使其能在更广泛的设备上运行。
3. **进行特定领域微调 (Fine-tuning)**：针对特定应用场景（如专业问答、文本摘要等），尝试对开源模型进行微调，以提升其在该领域的表现。
4. **探索模型可解释性与鲁棒性**：深入研究大语言模型的内部机制，理解其决策过程，并提升其在面对对抗性攻击或异常输入时的鲁棒性。
5. **关注伦理与偏见**：探讨大模型可能存在的伦理问题和偏见，并尝试寻找解决方案。

## 许可证 (License)

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢 (Acknowledgements)

- 感谢《人工智能导论》课程的老师，提供了宝贵的学习机会和指导。
- 感谢魔搭 (ModelScope) 平台，为本次大语言模型部署提供了稳定便捷的云计算环境。
- 感谢Qwen和ChatGLM项目团队，开源了如此优秀的大语言模型，极大地推动了人工智能领域的发展。

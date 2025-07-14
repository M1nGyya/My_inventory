import gradio as gr
import requests
import json
import os
import time

# FastAPI后端的地址
FASTAPI_BACKEND_URL = "http://127.0.0.1:8000/api/chat"
# 知识库目录
DATA_DIR = "./legal_data"

# 生成动态知识库列表的Md
def get_file_list_markdown():
    if not os.path.exists(DATA_DIR):
        return "知识库目录未找到。"

    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".pdf", ".docx"))]
    if not files:
        return "知识库为空。"

    md_text = f"本系统当前基于以下 **{len(files)}** 个文件进行问答:\n"
    for file in files:
        md_text += f"- `{file}`\n"
    return md_text


# 后端通信与 Gradio 事件处理函数
def get_bot_response(message, history):
    payload = {"query": message}
    try:
        response = requests.post(FASTAPI_BACKEND_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("answer", "抱歉，未能从后端获取有效回答。")
    except requests.exceptions.RequestException as e:
        return f"与后端通信时发生错误: {e}"
    except json.JSONDecodeError:
        return "后端返回了无效的数据格式。"

# 更新用户历史记录
def user_submit(user_message, history):
    return "", history + [[user_message, None]]

# 机器人回复
def bot_respond(history):
    user_message = history[-1][0]
    bot_message = get_bot_response(user_message, history)
    history[-1][1] = ""
    for char in bot_message:
        history[-1][1] += char
        time.sleep(0.01)
        yield history


# 欢迎语
def greet_on_load():
    welcome_message = "您好！我是您的法律咨询助手。请问有什么法律问题可以帮您？不要担心，尽管问吧。"
    return [[None, welcome_message]]


# --- Gradio 界面 ---
custom_css = """
/* 隐藏Gradio页脚 */
footer {display: none !important;}

/* 聊天气泡样式 */
#chatbot .user { background: #EBF5FF !important; border-radius: 10px !important; }
#chatbot .bot { background: #F0F0F0 !important; border-radius: 10px !important; }

/* 让文件列表在它的容器内居中 */
#file-list-container ul {
    list-style-position: inside;
    padding-left: 0;
    text-align: center;
}

/* 输入行内元素垂直居中，对齐发送按钮 */
#input-row { align-items: center; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), css=custom_css,
               title="法律智能咨询助手") as demo:
    # 主标题和描述
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>⚖️ 法律智能咨询助手</h1>
            <p>本系统基于本地法律文档和大语言模型构建，为您提供精准的法律信息问答。</p>
        </div>
        """
    )

    # 下拉栏
    with gr.Accordion("关于本系统 & 知识库来源", open=False):
        with gr.Column():
            gr.Markdown(
                """
                <div style="text-align: center;">
                    <h3>关于本项目</h3>
                    <p>本智能咨询系统基于大型语言模型（LLM）和本地法律知识库（RAG）构建，旨在提供初步的法律信息参考。</p>
                    <br>
                    <h3>知识库文件列表</h3>
                </div>
                """,
            )
            gr.Markdown(get_file_list_markdown(), elem_id="file-list-container")

    # 聊天机器人界面
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        show_label=False,
        bubble_full_width=False,
        height=550,
        avatar_images=(None, "https://img.icons8.com/plasticine/100/scales.png")
    )

    # 【已修改】输入区域的Row增加了elem_id
    with gr.Row(elem_id="input-row"):
        txt_input = gr.Textbox(
            show_label=False,
            placeholder="请在此输入您的法律问题...",
            scale=4,
        )
        submit_btn = gr.Button("发送", variant="primary", scale=1)

    # 按钮功能区
    with gr.Row():
        clear_btn = gr.ClearButton([txt_input, chatbot], value="✨ 新对话")
        gr.Examples(
            examples=[
                "消费者权益保护法都有哪些内容？",
                "试用期最长可以多久？",
                "我的公司倒闭了，我需要做什么？"
            ],
            inputs=txt_input,
            label = "示例问题"
        )

    # 绑定事件
    demo.load(greet_on_load, None, chatbot)

    submit_action = txt_input.submit(user_submit, [txt_input, chatbot], [txt_input, chatbot], queue=False).then(
        bot_respond, chatbot, chatbot
    )
    submit_btn.click(user_submit, [txt_input, chatbot], [txt_input, chatbot], queue=False).then(
        bot_respond, chatbot, chatbot
    )

# --- 4. 启动应用 ---
if __name__ == "__main__":
    print("启动 Gradio 前端应用...")
    demo.queue()
    demo.launch()
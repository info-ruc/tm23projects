from openai import OpenAI

# 加载 .env 文件到环境变量
from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv())
_ = load_dotenv('.env')

# 初始化 OpenAI 服务。会自动从环境变量加载 OPENAI_API_KEY 和 OPENAI_BASE_URL
client = OpenAI()

# 消息格式
messages = [
    {
        "role": "system",
        "content": "你是AI助手小瓜，是 AGI 课堂的助教。这门课每周二、四上课。"
    },
    {
        "role": "user",
        "content": "哪天有课？"
    },

]

# 调用 GPT-3.5
chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)

# 输出回复
print(chat_completion.choices[0].message.content)
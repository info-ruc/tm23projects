import uvicorn

from llmtuner import ChatModel, create_app


def main():
    chat_model = ChatModel()
    app = create_app(chat_model)
    uvicorn.run(app, host="1xxx.xxx.xxx.xxx", port=5237, workers=1)
    print("Visit http://xxx.xxx.xxx.xxx:5237/docs for API document.")


if __name__ == "__main__":
    main()

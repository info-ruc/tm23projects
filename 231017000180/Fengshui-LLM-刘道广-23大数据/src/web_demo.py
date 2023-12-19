from llmtuner import create_web_demo


def main():
    demo = create_web_demo()
    demo.queue()
    demo.launch(server_name="xxx.xxx.xxx.xxx", server_port=5278, share=False, inbrowser=True)


if __name__ == "__main__":
    main()

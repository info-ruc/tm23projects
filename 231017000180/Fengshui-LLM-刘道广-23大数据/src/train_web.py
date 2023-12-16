from llmtuner import create_ui


def main():
    demo = create_ui()
    demo.queue()
    demo.launch(server_name="xxx.xxx.xx.xx", server_port=5269, share=False, inbrowser=True)


if __name__ == "__main__":
    main()

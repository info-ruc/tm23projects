from datasets import load_dataset
dataset = load_dataset("wangrui6/Zhihu-KOL",cache_dir='E:\cache\dataset')
dataset.save_to_disk('E:\localstorage\dataset')
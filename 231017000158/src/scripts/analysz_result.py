import os.path
from config.config import *
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
def sort_fun(x):
    # 用自定义关系 将 x 映射 为 数值
    sort_dict = dict(zip(list("ynabcdefghi"),(1,2,3,4,5,6,7,8,9,10,11)))
    if not x:
        return 0
    elif x in ("oth","UTD"):
        return sort_dict.get(x,999)
    elif "\\" in x:
        xls = x.split("\\")
        xin = [sort_dict.get(y,99) for y in xls if y]
        return sum(xin)
    else:
        print(x)
        return 99
def save_pgn_4_analy(df,stem_name):
    # 计算两个字典的差异
    # df['Difference'] = df['stem_count_gold'] - df['stem_count_pred']
    # 绘制柱状图
    # ax = df.plot(kind='bar', color=['blue', 'green'], alpha=0.7)
    # print(df.columns)
    if "gold" not in df.index:
        ax = df.plot(kind='bar', color=['blue', 'green'], alpha=0.5)
    else:
        ax = df.plot(kind='bar', color='blue', alpha=1)
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    plt.xlabel('Key')
    plt.ylabel('Difference')
    plt.title('Comparison of Dictionaries')
    plt.savefig(os.path.join(results_dir_path, "图片", f"{stem_name}.png"))
    plt.show()
def main(is_train=True):
    results_path = os.path.join(results_dir_path,"结果对比.xlsx")
    # 1. 评估所有数据的准确率
    stem_compare_res_0 = pd.read_excel(results_path,usecols=["就诊流水号","填报数据项编码","选项或数据值_pre","选项或数据值_gold","是否正确"]).fillna("").astype(str)
    stem_compare_res = stem_compare_res_0.set_index(["就诊流水号"],inplace = False)
    stem_compare_res_stem_index = stem_compare_res_0.set_index(["填报数据项编码"],inplace=False)
    all_right_num = len([x for x in stem_compare_res["是否正确"].tolist() if x])
    all_num = len(stem_compare_res)
    count_res = Counter(stem_compare_res['选项或数据值_gold'].tolist())
    count_res = sorted(list(count_res.items()),key=lambda x:int(sort_fun(str(x[0]))))
    print(f"所有数据的数量：{all_num}，准确的数量：{all_right_num}，准确率：{all_right_num/all_num * 100:.2f}% !\n gold结果（排序）的统计信息为：{count_res}")
    # print(f"所有数据的结果分布：{count_res}")

    # 2. 每个stem的数据分布结果：
    stem_all = list(set(stem_compare_res["填报数据项编码"].tolist()))
    print(f"所有stem的数量为：{len(stem_all)}!")

    print(f"开始打印每个stem的统计信息--准确结果数量")
    stem_compare_res_stem_index["是否正确"]=stem_compare_res_stem_index["是否正确"].map(lambda x:1 if x else 0)
    stem_compare_res_stem_index = stem_compare_res_stem_index.groupby(stem_compare_res_stem_index.index)["是否正确"].sum()
    print(stem_compare_res_stem_index)
    for i in range(0,len(stem_all),10):
        print(stem_compare_res_stem_index[i:i+10])
        save_pgn_4_analy(stem_compare_res_stem_index[i:i+10],f"svery_stem_analy{i}-{i+10}")

    # 作图：每个stem的统计信息

    # 作图：所有的就诊的统计结果 ：
    stem_count_gold = Counter(
        stem_compare_res["选项或数据值_gold"].tolist())
    stem_count_pred = Counter(
        stem_compare_res["选项或数据值_pre"].tolist())
    print(f"所有的的stem_count_gold统计结果为：{stem_count_gold}\tstem_count_pred统计结果为：{stem_count_pred}")
    df = pd.DataFrame({'gold': stem_count_gold, 'pred': stem_count_pred})
    save_pgn_4_analy(df, "all")

    # 作图：每个stem的统计结果：
    for stem_name in stem_all:
        stem_count_gold = Counter(stem_compare_res["选项或数据值_gold"][stem_compare_res["填报数据项编码"]==stem_name].tolist())
        stem_count_pred = Counter(stem_compare_res["选项或数据值_pre"][stem_compare_res["填报数据项编码"]==stem_name].tolist())
        # print(f"{stem_name}的stem_count_gold统计结果为：{stem_count_gold}\tstem_count_pred统计结果为：{stem_count_pred}")
        df = pd.DataFrame({'gold': stem_count_gold, 'pred': stem_count_pred})
        # print(df)
        save_pgn_4_analy(df,stem_name)


    # stem_compare_res_2 =stem_compare_res_0.set_index(["就诊流水号","填报数据项编码"], inplace=False)
    # print(stem_compare_res_2)


    # for (vid,stem_name),res_dict in stem_compare_res_2.head().to_dict(orient="index").items():
    #     res_prd = res_dict.get('选项或数据值_pre')
    #     res_gold = res_dict.get('选项或数据值_gold')
    #     is_right = res_dict.get("是否正确")

    # 2. 评估每个stem的准确率

    # 3. 统计每个stem的数据分布情况
if __name__ == '__main__':
    main(is_train=True)
import json
import math
import os, sys
import re
from collections import defaultdict
import pandas as pd
import openpyxl

from config.config import *


def save_text_to_file(table_name_en, text, save_path):
    # if table_name_en != "discarge_note":
    #     return
    # 将text分段保存
    text_lines = text.split("\n")[3:]
    text_lines = [line for line in text_lines if line.strip() not in ["", "***"]]
    global sc_dir_name_all
    sc_text_dict = defaultdict(list)
    name_ker_words = ""
    for line in text_lines:
        # 二级标题的位置
        # if not re.search("^\s",line):   # pci_note 二级表名的特点，首字符非空
        if not re.search("^\s", line) and re.search(sc_table_dict_re[table_name_en], line.strip()):
            sc_dir_name_all.append(line.strip())  # 保存所有的sc_dir_name_all用于筛选二级表名
            res = re.search(sc_table_dict_re[table_name_en], line)
            if res:
                name_ker_words = res.group()
                name_ker_words = re.sub("[ \s\n]*]*", "", name_ker_words)
            else:
                raise Exception(f"未找到{line}对应的核心词")
        sc_text_dict[name_ker_words].append(line)
    with open(save_path, "w", encoding="utf8") as f:
        json.dump(sc_text_dict, f, ensure_ascii=False, indent=4)
    return sc_text_dict


def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        # 分析以上代码中的*path
        os.makedirs(os.path.join(*path))


def get_table_name_en(name_cn):
    for table_name_re_cn, table_name_en in doc_type_dict.items():
        if re.search(table_name_re_cn, name_cn):
            return table_name_en
    raise Exception(f"未找到{name_cn}对应的英文名")


def main():
    """数据预处理"""
    # 1. 读取数据
    name_ch_ls = []  # 查找所有数据中 表的类型——》用于转为英文，方便更新config中的doctype

    for _, _, files in os.walk(clinical_data_dir_path):
        # print(files)
        for fil in files:
            name_ch_ls.append(fil.split("_")[1])
    name_ch_ls = list(set(name_ch_ls))
    print(f"{len(name_ch_ls)}个table,分别是：{name_ch_ls}")

    # 2. 查找每个数据采集项的核心词出现在哪个表中：动脉支架
    for file in os.listdir(clinical_data_dir_path):
        # print(file)
        vid, table_name_cn, _, tid, _, tid_num = file.split("_")
        tid_num = tid_num.replace("txt", "")
        table_name_en = get_table_name_en(table_name_cn)

        # 入院时间：	2020年02月19日
        if file.endswith("txt"):
            with open(os.path.join(clinical_data_dir_path, file), encoding="utf8") as f:
                text = f.read()
                # 补充时间信息放在文件名中,默认是99999999
                time_info = "99999999"
                if re.search("入院",file):
                    res = re.search("入院.*?[：:]\s*(\d{4}.\d{1,2}.\d{1,2}.)", text)
                    if res:
                        time_info = re.sub("[^\d]*","",res.groups()[0])
                elif re.search("出院", file):
                    res = re.search("出院.*?[：:]\s*(\d{4}.\d{1,2}.\d{1,2}.)", text)
                    if res:
                        time_info = re.sub("[^\d]*", "", res.groups()[0])
                elif re.search("冠状动脉造影", file):
                    res = re.search("(\d{4}.\d{1,2}.\d{1,2}.)", text)
                    if res:
                        time_info = re.sub("[^\d]*", "", res.groups()[0])
                # 4. 将每个文档的分段保存在本地
                mkdir_if_not_exist([prepro_data_dir_path, vid])
                save_path = os.path.join(prepro_data_dir_path, vid, table_name_cn + tid_num+f"_{time_info}" + ".json")
                sc_text_dict = save_text_to_file(table_name_en, text, save_path)
                # print(f"已分层text并保存{file}到本地")
        else:
            print(f"{file}不是txt文件")

    print(f"已保存{clinical_data_dir_path}文件夹中的数据")
    # 3. 将每个文档进行分段
    #    1. 规则
    #    2. m3e 相似度



def main_4_check_stem():
    stem_info_xlsx_path = os.path.join(orig_data_dir_path, "STEMI数据项说明_备注说明_补充信息_v3.xlsx")
    columns_all = ["数据采集项", "数据类型", "备注", "备注3", "选项列表"]
    dfs = pd.read_excel(stem_info_xlsx_path, sheet_name="精简后", index_col="字段名称")[columns_all].fillna("")
    # dfs = pd.read_excel(stem_info_xlsx_path, sheet_name="精简后",index_col="字段名称")
    # 获取行索引的值
    row_indices = dfs.index
    row_indices = [x.strip() for x in row_indices if isinstance(x, str)]

    # 读取stem_info_xlsx_path文件的每个sheet的内容
    workbook = openpyxl.load_workbook(stem_info_xlsx_path)
    sheet_names = workbook.sheetnames
    for sheet_name in sheet_names:
        dfs = pd.read_excel(stem_info_xlsx_path, sheet_name=sheet_name)
        # 将缺失的sheet1内容转为字典形式
        if sheet_name == "Sheet1":
            pass


def data_process_4_physician_order():
    # 该封装函数放弃使用，合并在data_process_4_xlsx 中
    # 将 医嘱.xlsx文件 根据一级索引的方式 保存在本地vid.json:{医嘱优先级_医嘱类型：str结果拼接}
    order_path = os.path.join(orig_data_dir_path, "7-医嘱.xlsx")
    print(f"{order_path},{os.path.exists(order_path)}")
    dfs = pd.read_excel(order_path, index_col="就诊流水号")[["医嘱优先级", "医嘱类型", "医嘱名称", "医嘱状态"]].fillna(
        "")
    dfs_groups = dfs.groupby(dfs.index).apply(lambda x: x.to_dict(orient="records"))
    for vid in dfs_groups.index:
        mkdir_if_not_exist([prepro_data_dir_path, str(vid)])
        rows_list = dfs_groups.loc[vid]
        all_info_4_vid = defaultdict(str)
        for row_dict in rows_list:
            priority = row_dict.get("医嘱优先级")
            type_ = row_dict.get("医嘱类型")
            name = row_dict.get("医嘱名称")
            status = row_dict.get("医嘱状态")
            row_index = str(priority) + "_" + str(type_)
            if status:
                row_info = f'医嘱名称是{name}，其医嘱状态为{status}； '
            else:
                row_info = f'医嘱名称是{name}；'
            all_info_4_vid[row_index] += row_info
        with open(os.path.join(prepro_data_dir_path, str(vid), "医嘱.json"), "w", encoding="utf-8") as f:
            json.dump(all_info_4_vid, f, ensure_ascii=False, indent=4)

def data_process_4_other_results(file_name):
    file_path = os.path.join(orig_data_dir_path, file_name)
    other_results = pd.read_excel(file_path)
    print(f"{file_path},{os.path.exists(file_path)}")
    num_lines = len(other_results)
    vid_2_content = dict()
    for i in range(num_lines):
        data_dict = dict(other_results.loc[i])
        check_item = data_dict['检查项目']
        check_observation = data_dict['检查所见']
        check_conclusion = data_dict['检查结论']
        content = '检查项目：' + check_item + '\t检查所见：' + check_observation + '\t检查结论：' + check_conclusion + '\n'

        vid = str(data_dict['就诊流水号'])
        if vid not in vid_2_content:
            vid_2_content[vid] = ""
        vid_2_content[vid] += content

    for vid, content in vid_2_content.items():
        if content == "":
            content = "未进行胸片检查或其他检查。\n"

        with open(os.path.join(prepro_data_dir_path, str(vid), "其他检查结果.json"), "w", encoding="utf-8") as f:
            json.dump({"": content}, f, ensure_ascii=False, indent=4)

def data_process_4_ultrasonic_results(file_name):
    file_path = os.path.join(orig_data_dir_path, file_name)
    ultrasonic_results = pd.read_excel(file_path)
    print(f"{file_path},{os.path.exists(file_path)}")
    num_lines = len(ultrasonic_results)
    vid_2_content = dict()
    for i in range(num_lines):
        data_dict = dict(ultrasonic_results.loc[i])
        check_observation = data_dict['检查所见']
        if "检查结论" in data_dict:
            check_conclusion = data_dict['检查结论']
        elif "检查结论（超声大夫给出的诊断结论，不是临床诊断结果）" in data_dict:
            check_conclusion = data_dict['检查结论（超声大夫给出的诊断结论，不是临床诊断结果）']
        content = '超声心电图检查所见：\n' + check_observation + '\n检查结论：\n' + check_conclusion + '\n\n'

        vid = str(data_dict['就诊流水号'])
        if vid not in vid_2_content:
            vid_2_content[vid] = ""
        vid_2_content[vid] += content

    for vid, content in vid_2_content.items():
        with open(os.path.join(prepro_data_dir_path, str(vid), "超声心电图结果.json"), "w", encoding="utf-8") as f:
            json.dump({"": content}, f, ensure_ascii=False, indent=4)

def data_process_4_xlsx(file_name):
    # 将 医嘱.xlsx文件 根据一级索引的方式 保存在本地vid.json:{医嘱优先级_医嘱类型：str结果拼接}
    config_info = {"医嘱": {"index_col": ["医嘱优先级", "医嘱类型","医嘱开始时间","下医嘱时间"], "res_col": ["医嘱名称", "医嘱状态"]},
                   "其他检查结果": {"index_col": ["检查项目", "项目类别"], "res_col": ["检查所见", "检查结论"]},
                   "超声心动图结果": {"index_col": [],
                                      "res_col": ["检查所见", "检查结论（超声大夫给出的诊断结论，不是临床诊断结果）"]}}
    order_path = os.path.join(orig_data_dir_path, file_name)

    for re_ker, info in config_info.items():
        if re.search(re_ker, order_path):
            # print(order_path)
            break
    outfile_name = re_ker + ".json"
    index_col = info.get("index_col")
    res_col = info.get("res_col")
    print(f"{order_path},{os.path.exists(order_path)}")
    dfs = pd.read_excel(order_path, index_col="就诊流水号")[index_col + res_col].fillna("")
    # 针对时间的index ，修改值到年月日
    time_col = [x for x in index_col if re.search("时间",x)]
    if time_col:
        dfs[time_col[0]] = dfs[time_col[0]].map(lambda x:str(x)[:8] if x else "99999999")
        dfs[time_col[-1]] = dfs[time_col[-1]].map(lambda x:str(x)[:8] if x else "99999999")

    dfs_groups = dfs.groupby(dfs.index).apply(lambda x: x.to_dict(orient="records"))
    for vid in dfs_groups.index:
        mkdir_if_not_exist([prepro_data_dir_path, str(vid)])
        rows_list = dfs_groups.loc[vid]
        all_info_4_vid = defaultdict(str)
        for row_dict in rows_list:
            row_index, row_info = "", ""
            if index_col:
                for index_ in index_col:
                    row_index += row_dict.get(index_, "") + "_"

            for col in res_col:
                col_res = row_dict.get(col)
                if col_res:
                    row_info += f"{col}是{row_dict.get(col)}，"
            row_index = re.sub("[，；_]$", "", row_index)
            row_info = re.sub("[，；_]$", "", row_info)
            row_info += "； "
            all_info_4_vid[row_index] += row_info
        with open(os.path.join(prepro_data_dir_path, str(vid), outfile_name), "w", encoding="utf-8") as f:
            json.dump(all_info_4_vid, f, ensure_ascii=False, indent=4)


# 无意义词过滤器函数
def filter_stop_words(s):
    # words = s.split()
    # filtered_words = [word for word in words if word not in stop_words]
    s = re.sub("|".join(stop_words), " ", s)
    return re.sub(" +", " ", s)

def parser_stem_rule(rule):
    """解析每条stem数据项的规则"""
    # rule_4_pars = re.split(rule_2_ls_parser,rule)
    # rule_4_pars = [re.sub("\s","",x) for x in rule_4_pars if not re.search(rule_2_ls_parser,x) or x]
    # 1. 先获取对该结果的后处理方法
        # 入院第一天（不再用“首”代替入院第一天描述），围术期，时间首次（多个时间结果，选第一次），末次（多个匹配结果，选最后一次匹配到的结果）
    post_fun = ""
    post_fun_res = re.search("（(.*?)）【",rule)
    if post_fun_res:
        post_fun += post_fun_res.groups()[0]
        rule = re.sub("（.*?）(?=【)","",rule)
    # 2. 再获取行筛选正则条件
    kernals = ""
    kernals_res = re.search("\[(.*?)\]【",rule)
    if kernals_res:
        kernals += kernals_res.groups()[0]
        rule = re.sub("\[.*?\](?=【)","",rule)

    rule_2_parser = "(?P<first_layer>[\s\S]*?)\.(?P<sec_layer>[\s\S]*?)【(?P<pars_fun>[\s\S]*?)】[:：](?P<rule_info>[\s\S]*)"
    rule_ls = []
    try:
        res = re.findall(rule_2_parser,rule)
        # raise len(res) == 1
        if res:
            fst_layer, sec_layer, pars_fun, rule_info = res[0]
            return {"文件名": fst_layer, "表索引": sec_layer, "行筛选条件": kernals, "后处理方法":post_fun,"解析方式": pars_fun,
                    "规则": rule_info}
        else:
            print("医学规则没有识别出来", "rule=", rule, "res==", res)
            return {}
    except Exception as e:
        # print(e, "\n==",rl,"\n====",res)
        # return rule
        raise print(e, "rule=",rule,"res==",res)
    # rule_ls.append({"文件名":fst_layer,"表索引":sec_layer,"行筛选条件":kernals,"解析方式":pars_fun,"规则":rule_info})
def main_4_simi_resource():
    # 解析医学逻辑并保存
    stem_info_xlsx_path = os.path.join(orig_data_dir_path, stem_file)
    columns_all = ["数据采集项", "数据类型", "备注", "备注3", "选项列表"]
    dfs = pd.read_excel(stem_info_xlsx_path, sheet_name="精简后", index_col="字段名称")[columns_all].fillna("")
    # print(dfs.head())
    # dfs = pd.read_excel(stem_info_xlsx_path, sheet_name="精简后",index_col="字段名称")
    # 获取行索引的值
    row_indices = dfs.index
    row_indices = [x.strip() for x in row_indices if isinstance(x, str)]
    result_dict = {}
    stem_sim_all_info = defaultdict(dict)
    for row_index in row_indices:
        row_dict = dfs.loc[row_index].to_dict()
        source_table_ls = set()
        for key,value in list(row_dict.items()):
            if key.strip() =="备注3":
                values = re.split("[；;]",value)
                values = [re.sub("[;；]","",x.strip()) for x in values if x]
                parser_values = [parser_stem_rule(rule) for rule in values if rule]
                parser_values =[x for x in parser_values if x]
                row_dict["规则信息"] = parser_values
                del row_dict["备注3"]
        result_dict[row_index] = row_dict
        stem_table_info = defaultdict(set)
        if not type(row_dict["数据采集项"]) or type(row_dict["选项列表"]) == float:
            break
        """
        kernal_infos = row_dict["数据采集项"] + "\n" + row_dict["选项列表"] + "\n" + row_dict["备注2"]
        # 将该stem的核心语句 与 表-二级表信息 的内容 进行相似度匹配
        dir_ls = os.listdir(prepro_data_dir_path)
        for vid in dir_ls:
            table1_ls = os.listdir(os.path.join(prepro_data_dir_path, vid))
            for table1_name in table1_ls:
                with open(os.path.join(prepro_data_dir_path, vid, table1_name), "r", encoding="utf-8") as f:
                    table1_dict = json.load(f)
                    for table2_name, table2_info in table1_dict.items():
                        # sim_score1 = fuzz.token_set_ratio(kernal_infos,"\n".join(table2_info))
                        if not isinstance(table2_info, list):
                            table2_info = [table2_info]
                        # 将需要相似度匹配的文本进行 无意义过滤词处理
                        kernal_infos = filter_stop_words(kernal_infos)
                        table2_info = [filter_stop_words(s) for s in table2_info]
                        # sim_text,sim_score = process.extractOne(kernal_infos, table2_info,scorer=fuzz.WRatio,processor=filter_stop_words)  # 模糊匹配
                        sim_text, sim_score = process.extractOne(kernal_infos, table2_info)  # 模糊匹配

                        # text_sim_scores = process.extract(kernal_infos,table2_info)

                        # 通过相似度获 取对应的溯源表信息
                        table_name = table1_name[:-5] + "." + table2_name
                        table_name = re.sub("[:：、一二三四五六七八九十]", "", table_name)
                        if sim_score > 40:
                            print(f"{kernal_infos}\n\t{sim_score}\t{sim_text}")
                            source_table_ls.update({table_name})
                        stem_table_info[table_name].update({(sim_score, sim_text)})
        result_dict[row_index]["溯源表(自动)"] = list(source_table_ls)
        """
        stem_table_info = dict([(key, list(value)) for key, value in stem_table_info.items()])
        stem_sim_all_info[row_index] = stem_table_info

    os.makedirs(prepro_orig_data_dir_path, exist_ok=True)
    stem_info_dict_path = os.path.join(prepro_orig_data_dir_path, "stem_info_dict.json")
    with open(stem_info_dict_path, "w", encoding="utf8") as f:
        json.dump(stem_sim_all_info, f, ensure_ascii=False, indent=4)
    pd.DataFrame.from_dict(stem_sim_all_info, orient="index").to_excel(stem_info_dict_path.replace("json", "xlsx"),
                                                                       index=True)
    result_dict_path = os.path.join(prepro_orig_data_dir_path, "result_dict.json")
    pd.DataFrame.from_dict(result_dict, orient="index").to_excel(result_dict_path.replace(".json", ".xlsx"), index=True)
    with open(result_dict_path, "w", encoding="utf8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
    print(f"每个数据采集项的规则已解析，并保存在文件{result_dict_path}中！")

def main_4_get_fisrt_admit_time():
    # 根据就诊目录下转为二级标题的json数据，获得首次如愿时间
    vids = os.listdir(prepro_data_dir_path)
    fisrt_admit_time = ""
    for vid in vids:
        operation_time = []
        for file in os.listdir(os.path.join(prepro_data_dir_path,vid)):
            # 1. 入院记录中的入院时间
            if re.search("入院记录",file):
                with open( os.path.join(prepro_data_dir_path,vid,file), "r",encoding="utf-8") as f:
                    file_json = json.load(f)
                    for key,value in file_json.items():
                        if re.search("入院记录",key):
                            res = re.search("入院时间[:：\s\\t]*(\d{4})年(\d{2})月(\d{2})日","".join(value))
                            if res:
                                fisrt_admit_time = "".join(list(res.groups()))
            # 2. 医嘱中的首次时间
            elif re.search("医嘱",file):
                with open(os.path.join(prepro_data_dir_path, vid, file), "r", encoding="utf-8") as f:
                    file_json = json.load(f)
                    process_time = sorted([int(x.split("_")[-2]) for x in list(file_json.keys())])
                    if not fisrt_admit_time:
                        fisrt_admit_time = str(process_time[0])
                    operation_time.extend([x.split("_")[-2:] for x in list(file_json.keys()) if re.search("手术",x)])
        if fisrt_admit_time or operation_time:
            with open(os.path.join(prepro_data_dir_path,vid,"补充信息.json"),"w",encoding="utf-8") as f:
                json.dump({"首次入院时间":fisrt_admit_time,"围术期":operation_time},f,ensure_ascii=False,indent=4)
    print("每个就诊的首次入院时间信息已补充！")
if __name__ == '__main__':
    sc_dir_name_all = []
    # main()
    # sc_dir_name_all = list(set(sc_dir_name_all))
    # print(f"sc_dir_name_all:{len(sc_dir_name_all)}:{sc_dir_name_all}")

    # 以下将xlsx文档数据转为两级结构数据json中
    # data_process_4_xlsx("5-超声心动图结果.xlsx")
    # data_process_4_xlsx("6-其他检查结果.xlsx")
    # data_process_4_xlsx("7-医嘱.xlsx")
    # data_process_4_ultrasonic_results("5-超声心动图结果.xlsx")
    # data_process_4_other_results("6-其他检查结果.xlsx")
    main_4_simi_resource()
    # main_4_get_fisrt_admit_time()


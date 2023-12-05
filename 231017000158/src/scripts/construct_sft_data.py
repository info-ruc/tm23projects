import re

import json

import os
from config.config import *
from test_4_model_prompt import get_all_stem_info, get_check_vids_info, get_cli_info_4_vid, get_context_info_4_vid_4_stem
import pandas as pd

def get_gold_ans(gold_annotaion_path):
    gold_ann = pd.read_excel(gold_annotaion_path)
    num_lines = len(gold_ann)
    vid_2_stem_ans = dict()
    for i in range(num_lines):
        data_dict = dict(gold_ann.loc[i])
        vid = data_dict["就诊流水号"]
        vid = str(vid)
        stem = data_dict["填报数据项编码"]
        ans = data_dict["选项或数据值"]
        assert isinstance(ans, str) or pd.isna(ans), data_dict
        if vid not in vid_2_stem_ans:
            vid_2_stem_ans[vid] = dict()
        vid_2_stem_ans[vid][stem] = ans
        # print(ans)
    return vid_2_stem_ans

def convert_choice_to_phrase(choice, prompt):
    mapping_rule = "(?P<a>A\.)|(?P<b>B\.)|(?P<c>C\.)|(?P<d>D\.)|(?P<UTD>E\.)"
    for line in prompt.split('\n'):
        res = re.finditer(mapping_rule, line)
        for x in res:
            for group_name, group_value in x.groupdict().items():
                if group_value and group_name == choice:
                    return line
    return None

def get_sample_4_sft(file_re_info, rule_info, context_4_stem_vid, gold_ans, vid, stem_name):
    """
    根据对现有27-48中的stem的规则发现，只有三个关于”禁忌症“的字段需要模型进行问答，以此作为特征构建prompt。
    :param file_re_info: 文件名信息，即一级索引
    :param sec_index_re: 表索引信息，即二级信息
    :param stem_cn_name: stem中文名
    :param rule_info: prompt
    :param context_4_stem_vid: 该就诊中关于该stem的相关病历文本信息
    :return:
    """
    rule_info = rule_info.replace("\\n", "\n")
    context_4_stem_vid = context_4_stem_vid.strip()
    instruction = f"你将阅读一段来自{file_re_info}的病历文本，并根据病历内容回答一个问题。\n病历文本：\n{context_4_stem_vid}\n根据病历内容，请问{rule_info}"
    # instruction = f"{context_4_stem_vid}"

    def print_invalid_sample():
        # print(f"标注文件中就诊{vid}的数据项{stem_name}值为{gold_ans}，无效")
        # print("*" * 100)
        # print(instruction)
        # print("*" * 100)
        # print()
        return

    if pd.isna(gold_ans):
        if stem_name in do_not_ans_when_utd:
            gold_ans = '\"UTD\"'
        else:
            print_invalid_sample()
            return None
    
    assert isinstance(gold_ans, str)

    # # 只查看有禁忌的病例数据
    # if ("禁忌" not in rule_info) or gold_ans[1:-1] != 'y':
    #     return None
    
    target = ''
    gold_ans = gold_ans[1:-1]
    if gold_ans == 'y':
        if len(context_4_stem_vid) < 10:
            return None

        target = '是'
    elif gold_ans =='n':
        target = '否'
    else:
        # 单选
        if gold_ans[0] != '\"':
            assert stem_name != 'STEMI-3-2-3-1'
            assert gold_ans in ['a', 'b', 'c', 'd', 'UTD'], gold_ans
            if gold_ans != 'UTD' and len(context_4_stem_vid) < 10:
                print_invalid_sample()
                return None
            target = convert_choice_to_phrase(gold_ans, rule_info)
        # 多选
        else:
            assert stem_name == 'STEMI-3-2-3-1'
            # print(gold_ans)
            phrases = []
            for choice in gold_ans.split(","):
                assert choice[0] == '\"'
                choice = choice[1:-1]
                assert choice in ['a', 'b', 'c', 'd', 'UTD'], choice
                phrase = convert_choice_to_phrase(choice, rule_info)
                assert phrase is not None
                phrases.append(phrase)
            if gold_ans != '\"UTD\"' and len(context_4_stem_vid) < 10:
                print_invalid_sample()
                return None
            target = '\n'.join(phrases)

    sample = {
        "instruction": instruction,
        "output": target,
        "history": None,
    }
    return sample

def main():
    # 1. 读取stem的配置信息
    stem_info_dict = get_all_stem_info()
    # 2. 读取就诊流水号信息
    all_vids = get_check_vids_info()
    # 从标注文件获取每个就诊的各个数据项对应的答案
    vid_2_stem_answer = get_gold_ans(train_gold_annotaion_path)
    sft_data_samples = []
    # 3. 生产每个就诊的每个stem问题结果
    for vid in all_vids:
        vid_file_path = os.path.join(train_prepro_data_dir_path, vid)
        files_4_vid = os.listdir(vid_file_path)
        # 3.1 加载该就诊下的所有转化格式后的病历信息
        cli_info_4_vid = get_cli_info_4_vid(vid_file_path,files_4_vid)
        # 3.2 依次获得每个stem的结果，需要根据有向无环图的顺序获取结果
        for stem_name,stem_info in stem_info_dict.items():
            stem_cn_name = stem_info["数据采集项"]
            stem_type = stem_info["数据类型"]
            stem_other_info = stem_info["备注"]
            stem_select_info = stem_info["选项列表"]
            stem_rule_info = stem_info["规则信息"]
            try:
                gold_ans = vid_2_stem_answer[vid][stem_name]
            except:
                print(f"标注文件中不包含就诊{vid}的数据项{stem_name}\n")
                continue
            
            # 3.3 通过stem信息获得结果
            for stem_info in stem_rule_info:
                if isinstance(stem_info,dict):
                    file_re_info = stem_info["文件名"].strip()
                    sec_index_re = stem_info["表索引"].strip()
                    line_re = stem_info["行筛选条件"].strip()
                    parser_fun = stem_info["解析方式"].strip()
                    rule_info = stem_info["规则"].strip()
                    # 3.4 获取该就诊关于该stem的相关病历内容。
                    context_4_stem_vid, _ = get_context_info_4_vid_4_stem(file_re_info, sec_index_re, cli_info_4_vid,line_re,stem_cn_name)

                    if parser_fun != "模型":
                        continue
                    sample = get_sample_4_sft(file_re_info, rule_info, context_4_stem_vid, gold_ans, vid, stem_name)
                    if sample:
                        # sample["数据采集项"] = stem_cn_name
                        sft_data_samples.append(sample)
                else:
                    raise print(f"{vid}就诊中{stem_name}:{stem_cn_name}的stem_info应该解析为dict，但是实际为{stem_info}，错误！")
    
    save_sft_data_path = "sft_data.json"
    with open(save_sft_data_path, 'w') as f:
        json.dump(sft_data_samples, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()

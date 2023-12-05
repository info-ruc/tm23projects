import re

import json

import os
from config.config import *
import pandas as pd
import queue
from copy import deepcopy
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_model_and_tokenizer(model_path):
    if 'chatglm' in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()
    elif 'qwen' in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, fp16=True).eval()
    else:
        raise ValueError(model_path)
    return model, tokenizer

# 有向无环图中各结点的入度，以及每个结点指向的所有结点
def prepare_degs_and_tails(dependencies):
    degs = dict()
    tails = dict((k, []) for k in dependencies)
    for k, v in dependencies.items():
        degs[k] = len(v)
        for item in v:
            tails[item].append(k)
    return degs, tails
# 拓扑排序
def toposort(degs, tails):
    res = []
    q = queue.Queue()
    for term, in_deg in degs.items():
        if in_deg == 0:
            q.put(term)
    while q.qsize() > 0:
        front = q.get()
        res.append(front)
        for term in tails[front]:
            degs[term] -= 1
            if degs[term] == 0:
                q.put(term)
    return res

# 根据所依赖数据项的预测值判断是否需要填写当前数据项
def check_whether_ans_via_dependency(term, dependent_terms, stem_2_answer, is_dead):
    flag_must_ans = False
    flag_may_ans = True
    flag_need_not_ans = False
    if term in do_not_ans_if_dead and is_dead == True:
        flag_need_not_ans = True
    for dependent_term in dependent_terms:
        depen_term_ans =  stem_2_answer[dependent_term]
        if (term, dependent_term) in must_answer and depen_term_ans == must_answer[(term, dependent_term)]:
            flag_must_ans = True
        if (term, dependent_term) in may_answer and depen_term_ans != may_answer[(term, dependent_term)]:
            flag_may_ans = False
        if (term, dependent_term) in need_not_answer and depen_term_ans == need_not_answer[(term, dependent_term)]:
            flag_need_not_ans = True
        if (term, dependent_term) in need_not_answer_rev and need_not_answer_rev[(term, dependent_term)] not in depen_term_ans:
            flag_need_not_ans = True
    # 当模型预测值为"UTD"时不需要填的字段
    if term in do_not_ans_when_utd and stem_2_answer[term] == 'UTD':
        flag_need_not_ans = True
    # 默认要填字段（除非和依赖有冲突）
    if flag_may_ans and not flag_need_not_ans and term in must_ans_stems:
        flag_must_ans = True
    return flag_must_ans, flag_may_ans, flag_need_not_ans
# 根据有向无环图进行后处理
def post_processing(vid_2_stem_answer, vid_2_is_dead):
    new_vid_2_stem_answer = deepcopy(vid_2_stem_answer)
    for vid, stem_2_answer in vid_2_stem_answer.items():
        new_stem_2_ans = deepcopy(stem_2_answer)
        for stem, ans in stem_2_answer.items():
            # 该数据项所依赖的所有数据项
            depen_terms = term_dependencies[stem]
            must_ans, may_ans, need_not_ans = check_whether_ans_via_dependency(stem, depen_terms, new_stem_2_ans, vid_2_is_dead[vid])
            # 必须填写
            if must_ans:
                # assert may_ans == True and need_not_ans == False, f"{stem}的依赖项预测结果{[(x, new_stem_2_ans[x]) for x in depen_terms]}矛盾"
                if not ans:
                    print(f"[{vid}][Conflict][Must answer]:"
                          f"{stem}无预测值，与其依赖项的预测结果{[(x, new_stem_2_ans[x]) for x in depen_terms]}矛盾，重置为n")
                    new_stem_2_ans[stem] = stem_2_default_value[stem] if stem in stem_2_default_value else 'n'
                    new_vid_2_stem_answer[vid][stem] = stem_2_default_value[stem] if stem in stem_2_default_value else 'n'
            # 不需要填写
            if not may_ans or need_not_ans:
                if ans:
                    print(f"[{vid}][Conflict][Need not answer]:"
                          f"{stem}有预测值{ans}，与其依赖项的预测结果{[(x, new_stem_2_ans[x]) for x in depen_terms]}矛盾，重置为空")
                new_stem_2_ans[stem] = ""
                new_vid_2_stem_answer[vid][stem] = ""
            res = new_vid_2_stem_answer[vid][stem]
            if stem == "STEMI-3-2-3-1" and res:
                # new_vid_2_stem_answer[vid][stem] = json.dumps(new_vid_2_stem_answer[vid][stem].split('\\'))
                new_vid_2_stem_answer[vid][stem] = "[" + ",".join(['\"'+x+'\"' for x in res.split('\\')]) + "]"
            elif res:
                new_vid_2_stem_answer[vid][stem] = res.split("\\")[0]
    return new_vid_2_stem_answer

def get_result_4_re(rule_info, context_4_stem_vid):
    # 基于正则的方式获取结果
    res_dict = {}
    # print(rule_info)
    try:
        res = re.finditer(rule_info, context_4_stem_vid)
    except:
        print(rule_info)
        exit(0)
    if res:
        for x in res:
            for group_name,group_value in  x.groupdict().items():
                if group_value:
                    res_dict[group_name] = group_value
    else:
        return ""
    res_group_name = [re.sub("\d+","",x) for x in list(res_dict.keys()) if x]
    res_group_name = sorted(list(set(res_group_name)))
    return "\\".join(res_group_name)

def get_result_4_model(file_re_info, sec_index_re, stem_cn_name, rule_info, context_4_stem_vid, model, tokenizer):
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
    context_4_stem_vid = context_4_stem_vid.strip()[:2500]
    prompt = f"你将阅读一段来自{file_re_info}的病历文本，并根据病历内容回答一个问题。\n病历文本：\n{context_4_stem_vid}\n根据病历内容，请问{rule_info}"
    r, h = model.chat(tokenizer, prompt, history=[], do_sample=False)
    print("*"*100)
    print(prompt)
    print()
    print(r)
    print("*"*100)
    result = parse_model_answer(r, stem_cn_name)
    return result
def parse_model_answer(response, term):
    # 对模型回复进行解析得到答案
    # if response in ['是', '否']:
    #     return yes_or_no_mapping[response]

    res_dict = {}
    mapping_rule = "(?P<y>是)|(?P<n>否)|(?P<a>A\.)|(?P<b>B\.)|(?P<c>C\.)|(?P<d>D\.)|(?P<UTD>E\.)"
    res = re.finditer(mapping_rule, response)
    if res:
        for x in res:
            for group_name,group_value in x.groupdict().items():
                if group_value:
                    res_dict[group_name] = group_value
    else:
        print(response)
        exit(0)
    res_group_name = [re.sub("\d+","",x) for x in list(res_dict.keys()) if x]
    res_group_name = sorted(list(set(res_group_name)))
    return "\\".join(res_group_name)
def check_stem_res_and_type(stem_type, stem_res):
    if stem_type == "字符串":
        assert len(stem_res.split("\\")) == 1
    elif stem_type == "数组":
        pass
    else:
        raise print("数据采集项的数据类型既不是字符串也不是数组，不符合要求！")
def combine_stem_res_4_rule_and_model(stem_results):
    # stem_results：[{规则：答案},{模型：答案}]
    # 若既有规则又有模型，获取答案的依据:
    #   all没有冲突，以唯一答案为准；
    #   all有冲突，以规则答案为准；若规则答案多个，则以y为准，；是以规则为准==》若规则答案冲突，则以y为准 todo
    all_res = list(set([ans for xdict in stem_results for rl,ans in xdict.items() if ans]))
    rule_res = list(set([ans for xdict in stem_results for rl,ans in xdict.items() if rl == "规则" and ans]))
    model_res = list(set([ans for xdict in stem_results for rl,ans in xdict.items() if rl == "模型" and ans]))
    if not all_res:
        return ""
    if len(all_res) == 1:
        return all_res[0]
    elif len(rule_res) == 1:
        return rule_res[0]
    elif len(model_res) == 1:
        return model_res[0]
    else:
        if "y" in rule_res:
            return "y"
        else:
            return "n"

def get_precondition_2_secect_line(stem_cn_name,condition_info):
    """

    :param stem_cn_name:
    :param condition_info:
    :return:
    """
    if re.search("首次",stem_cn_name):
        return condition_info.get("首次入院时间","")
    elif re.search("围术期",stem_cn_name):
        terms = condition_info.get("围术期",[])
        return terms
        # for term in terms:
        #     for start_time,end_time in term:
        #         start_time = int(start_time) -1
        #         end_time = int(end_time) +1
        #         return
    else:
        return ""
def is_filter_4_sec_index(precondition_re,sec_index):
    if isinstance(precondition_re, str):
        if re.search(precondition_re, sec_index):
            return False
    else:
        for start_time, end_time in precondition_re:
            # try:
            if int(start_time) -1 <= int(sec_index.split("_")[-2]) <= int(end_time) +1:
                return False
            # except:
            #     print(sec_index, start_time, end_time)
            #     exit(0)
    return True
def get_context_info_4_vid_4_stem(file_re_info,sec_index_re,cli_info_4_vid,line_re,stem_cn_name):
    # stem_cn_name: 如果是“首”相关，则医嘱规则增加 首次入院时间和sec_index中医嘱开始相同相同；
    #                 如果是“围术期”,则医嘱规则增加 医嘱开始和结束时间限制下 sec_index筛选条件
    context,context_4_file_name = "",defaultdict(list)
    # 获取前置条件
    precondition_re = get_precondition_2_secect_line(stem_cn_name,cli_info_4_vid.get("补充信息",{}))
    for file_name, cli_info in cli_info_4_vid.items():
        if re.search(file_re_info, file_name):
            for sec_index, sec_info in cli_info.items():
                if precondition_re and re.search("医嘱",file_re_info):
                    if is_filter_4_sec_index(precondition_re, sec_index):
                       continue
                # sec_index = re.sub("_\d{4,}","",sec_index)  # 删除掉时间信息
                if re.search(sec_index_re, sec_index):
                    # 通过line_re行筛选依据，过滤无用信息
                    if isinstance(sec_info,list):
                        line_str = "\n".join([x for x in sec_info if re.search(line_re,x)])
                    else:
                        line_str = sec_info if re.search(line_re,sec_info) else ""
                    context += line_str + '\n'
                    if line_str:
                        context_4_file_name[file_name+"_"+sec_index].append(line_str)
    return context,context_4_file_name
def get_all_stem_info():
    result_dict_path = os.path.join(prepro_orig_data_dir_path, "result_dict.json")
    with open(result_dict_path,"r",encoding="utf-8") as f:
        stem_info_dict = json.load(f)
    # 有向无环图
    degs, tails = prepare_degs_and_tails(term_dependencies)
    # 拓扑排序
    term_seq = toposort(degs, tails)
    assert len(term_seq) == len(stem_info_dict)
    new_stem_info_dict = {}
    for stem_name in term_seq:
        new_stem_info_dict[stem_name] = stem_info_dict[stem_name]
    return new_stem_info_dict
def get_check_vids_info():
    # 读取就诊列表中的就诊id信息，并核对和解析的数据中vid是否相同
    with open(os.path.join(orig_data_dir_path,"3-就诊流水号列表.txt"),"r",encoding="utf-8") as f:
        all_vids = f.readlines()
    all_vids = [x.strip() for x in all_vids if x]
    prepro_vids = os.listdir(prepro_data_dir_path)
    assert len(all_vids) == len(prepro_vids)
    return all_vids
def get_cli_info_4_vid(vid_file_path,files_4_vid):
    # 读取该就诊下的所有病历信息
    cli_info_4_vid = {}
    for file_4_cli_info in files_4_vid:
        with open(os.path.join(vid_file_path, file_4_cli_info), "r", encoding="utf-8") as f:
            cli_info = json.load(f)
        cli_info_4_vid[file_4_cli_info[:-5]] = cli_info
    return cli_info_4_vid

def covert_dict_2_pd(vid_2_stem_answer):
    new_res = []
    for vid,stem_info in vid_2_stem_answer.items():
        for name,res in stem_info.items():
            # assert isinstance(res, str)
            # if name == "STEMI-3-2-3-1":
            #     # res = "[" + ",".join(['\"'+str(x)+'\"' for x in res.split('\\')]) + "]"
            #     res = res.split('\\')
            new_res.append({"就诊流水号":vid,	"填报数据项编码":name,	"选项或数据值":res})
    return new_res

def compare_results(vid_2_stem_answer, gold_annotaion_path):
    compare_res,stem_names_pre,stem_names_gold = [],set(),set()
    vid_2_stem_gold =pd.read_excel(gold_annotaion_path,usecols=["就诊流水号", "填报数据项编码", "选项或数据值"]).fillna("").astype(str)
    vid_2_stem_gold.set_index(["就诊流水号", "填报数据项编码"],inplace = True)
    new_vid_2_stem_gold = defaultdict(dict)
    for (vid,name),res_gold in list(vid_2_stem_gold.to_dict().values())[0].items():
        new_vid_2_stem_gold[vid].update({name:re.sub("^\"|\"$","",res_gold)})
        stem_names_gold.update({name})
    eq_num = 0
    for vid,stem_info_pre_4_vid in vid_2_stem_answer.items():
        stem_info_gold_4_vid = new_vid_2_stem_gold.get(vid,{})
        for name,value_pred in stem_info_pre_4_vid.items():
            stem_names_pre.update({name})
            value_gold = stem_info_gold_4_vid.get(name,"")
            is_equal = 1 if value_pred == value_gold else ""
            eq_num +=1 if is_equal else 0
            compare_res.append({"就诊流水号":vid, "填报数据项编码":name, "选项或数据值_pre":value_pred,"选项或数据值_gold":value_gold,"是否正确":is_equal})
    pre_res_all = [x.get("选项或数据值_pre") for x in compare_res if x.get("选项或数据值_pre")]
    gold_res_all = [x.get("选项或数据值_gold") for x in compare_res if x.get("选项或数据值_gold")]
    # print("pre_res_all",pre_res_all)
    # print("gold_res_all",gold_res_all)
    # print(stem_names_pre)
    # print(stem_names_gold)
    print(f"pre的数据采集项数量为{len(stem_names_pre)}，比比赛多的为{stem_names_pre-stem_names_gold}；gold的数据数据采集项数量为{len(stem_names_gold)}，比48个多的为{stem_names_gold-stem_names_pre}！")
    print(f"pre非空结果数量{len(pre_res_all)}，gold非空结果数量{len(gold_res_all)}。准确率：{eq_num}/{len(compare_res)}，{eq_num/len(compare_res) * 100:.2f}%  !")
    pd.DataFrame.from_records(compare_res).to_excel(os.path.join(results_dir_path,"结果对比.xlsx"))

def post_fun_4_one_stem(stem_type,post_fun,stem_res_dict,cli_info_4_vid):
    """对每条医学规则的匹配结果的后处理(此处的后处理 是对文件时间【即文件名上的时间】进行对比)
    post_fun:对结果进行后处理操作，例如：
    入院第一天（stem_name中若有“首”，则同义，不需要再写，默认要求入院第一天），围术期，时间首次（多个时间结果，选第一次），末次（多个匹配结果，选最后一次匹配到的结果）
    stem_res_dict：经过正则和前处理 之后的答案：{file_name: "a\\b"}
    cli_info_4_vid：每个就诊抽取出来的综合信息，例如{首次入院时间: ，围术期:[(start_time,end_time),()]}
    	入院第一天（不再用“首”代替入院第一天描述）
	围术期，
	时间首次（多个时间结果，选第一次），例如pci记录时间有多个
	最小（）
	最大（）
    """
                            
    # if not stem_res_dict:
    #     if re.search("默认", post_fun):   # 只有答案为空，才有默认值
    #         return  re.search("(?<=默认).", post_fun).group()
    # 文件所有答案的集合
    res_all_ls = list(set([y for x in list(stem_res_dict.values()) for y in x.split("\\") if y]))
    other_info = cli_info_4_vid.get("补充信息",{})
    fist_admission_time = other_info.get("首次入院时间","")
    wsq_time = other_info.get("围术期",[])
    res_ls = []
    def sort_key_fun(x, num=0):
        # "UTD","oth"的排序建为0，表明先排序，为>1,表明比其他后排序
        en_sort = dict(zip(("a","b","c","d","e","f","g","h"),(1,2,3,4,5,6,7,8,9,10)))
        if x in ("UTD", "oth"):
            return num
        else:
            return en_sort.get(x)
    # 1. 根据文件名（时间信息）筛选合适答案
    # 1.1 只获取入院第一天的 信息
    for file_name,stem_res_4_file in stem_res_dict.items():
        time_info = file_name.split("_")[-1]
        if re.search("入院第一天",post_fun):
            # print(f"{time_info}==>{fist_admission_time}")
            if time_info == fist_admission_time:
                res_ls.append(stem_res_4_file)
    # 1.3 只获取在围术期 范围内的文件时间的信息
    if re.search("围术期",post_fun) and stem_res_dict:
        for tm, res in stem_res_dict.items():
            # print(f"tm:{tm},wsq_time:{wsq_time}.stem_res_dict:{stem_res_dict}")
            res_ls.append(res) if not is_filter_4_sec_index(wsq_time,tm+"_0") else ""
    # 1.2 只获取所有文件时间中最早的答案
    if re.search("时间首次",post_fun) and stem_res_dict:
        # print(f"stem_res_dict{stem_res_dict}")
        if res_ls:
            new_stem_res_dict = dict([(x,y) for x ,y in stem_res_dict.items() if y in res_ls])
        else:
            new_stem_res_dict = stem_res_dict
        if [x for x in list(new_stem_res_dict.keys()) if re.search("医嘱",x)]:
            return sorted(list(new_stem_res_dict.items()), key=lambda x: x[0].split("_")[-2])[0][1]
        else:
            return sorted(list(new_stem_res_dict.items()), key=lambda x: x[0].split("_")[-1])[0][1]

    # 当以上匹配规则都没有的话，说明所有答案都需要
    if re.search("入院第一天|时间首次|围术期",post_fun):
        if not res_ls:
            # if re.search("默认", post_fun):   # 只有答案为空，才有默认值
            #     return re.search("(?<=默认).", post_fun).group()
            # else:
            return ""
        else:
            # 经过1. 筛选之后的答案为需要后续处理的所有答案
            res_ls = sorted([x for y in res_ls for x in y.split("\\")])
    else:
        res_ls = sorted(res_all_ls)
    # print(f"文件时间筛选之后的答案：{res_ls}")
    # 2. 根据答案所有信息匹配答案
    if re.search("最小",post_fun):
        # 答案默认a最小, 有a-z结果就不要oth和UTD等其他结果
        res_ls=sorted(res_ls,key=lambda x: sort_key_fun(x,99))[0]
    elif re.search("最大",post_fun):
        # 答案默认a最小，则选择排序中最大的结果
        res_ls = sorted(res_ls,key=lambda x: sort_key_fun(x,0))[-1]

    return "\\".join(res_ls)
    # if len(stem_res_dict) >= 2:
    #     if stem_type == "数组":  # 结果有多个：STEMI-3-2-3-1	主要病变血管
    #         stem_res_4_rule = "\\".join([y for x, y in stem_res_dict.items()])
    #     else:
    #         # 当结果有多个，但是数据类型是“字符串”时，根据时间，获取时间最早的那个
    #         min_time_res = sorted(list(stem_res_dict.items()), key=lambda x: x[0][:-5].split("_")[-1])[0]
    #         stem_res_4_rule = list(min_time_res.values())[0]
    # else:
    #     stem_res_4_rule = list(stem_res_dict.values())[0]
    # return stem_res_4_rule
def main():
    # 0. 加载大语言模型和tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)

    # 1. 读取stem的配置信息
    new_stem_info_dict = get_all_stem_info()
    # 2. 读取就诊流水号信息
    all_vids = get_check_vids_info()
    # 对于每个数据项，各个就诊流水号对应的预测答案
    vid_2_stem_answer = dict((vid, {}) for vid in all_vids)
    vid_2_is_dead = dict()  # TODO
    # 3. 生产每个就诊的每个stem问题结果
    for idx, vid in enumerate(all_vids):
        if is_train and idx % 15 != 0:
            continue
        vid_file_path = os.path.join(prepro_data_dir_path,vid)
        files_4_vid = os.listdir(vid_file_path)
        # 3.1 加载该就诊下的所有转化格式后的病历信息
        cli_info_4_vid = get_cli_info_4_vid(vid_file_path,files_4_vid)
        # TODO
        vid_2_is_dead[vid] = '死亡记录' in cli_info_4_vid

        # 3.2 依次获得每个stem的结果，需要根据有向无环图的顺序获取结果
        for stem_name,stem_info in new_stem_info_dict.items():
            stem_cn_name = stem_info["数据采集项"]
            stem_type = stem_info["数据类型"]
            stem_other_info = stem_info["备注"]
            stem_select_info = stem_info["选项列表"]
            stem_rule_info = stem_info["规则信息"]
            stem_results = []

            # 对于某些数据项，可以根据其依赖的数据项的预测值直接得到答案（例如STEMI-3-2-1为y则STEMI-3-2-2为n）
            if stem_name in infer_answer_via_dependency:
                for depen_stem, depen_ans_mapping in infer_answer_via_dependency[stem_name].items():
                    assert depen_stem in vid_2_stem_answer[vid]
                    depen_stem_answer = vid_2_stem_answer[vid][depen_stem]
                    if depen_stem_answer in depen_ans_mapping:
                        inferred_stem_res = depen_ans_mapping[depen_stem_answer]
                        # print(f"inferred_stem_res: {inferred_stem_res}")
                        stem_results.append({"规则": inferred_stem_res})

            # 3.3 通过stem信息获得结果
            for stem_info in stem_rule_info:
                if isinstance(stem_info,dict):
                    file_re_info = stem_info["文件名"].strip()
                    sec_index_re = stem_info["表索引"].strip()
                    line_re = stem_info["行筛选条件"].strip()
                    post_fun = stem_info["后处理方法"].strip()
                    parser_fun = stem_info["解析方式"].strip()
                    rule_info = stem_info["规则"].strip()
                    # 3.4 获取该就诊关于该stem的相关病历内容。（基于正则和“首”，”围术期“的条件前置）
                    context_4_stem_vid,context_4_file_name = get_context_info_4_vid_4_stem(file_re_info, sec_index_re, cli_info_4_vid,line_re,stem_cn_name)
                    if not context_4_stem_vid:
                        stem_results.append({"规则":""})
                    # 3.5 根据规则或模型，获取相应答案，并使答案符合比赛要求
                    elif parser_fun == "规则":
                        stem_res_dict = {}
                        for file_name,context_4_file in context_4_file_name.items():
                            stem_res_4_rule_ = get_result_4_re(rule_info,context_4_stem_vid)
                            if stem_res_4_rule_:
                                stem_res_dict.update({file_name:stem_res_4_rule_})
                        if post_fun:
                            # 3.4.1 对每条规则的匹配结果的后处理操作：  # todo
                            # 若有多个文件的病历文本符合条件，对所有结果综合处理
                            stem_res_4_rule = post_fun_4_one_stem(stem_type,post_fun,stem_res_dict,cli_info_4_vid)
                        else:
                            stem_res_4_rule = "\\".join(list(set(list(stem_res_dict.values()))))
                        stem_res_4_rule="\\".join(sorted(list(set(stem_res_4_rule.split("\\")))))
                        # check_stem_res_and_type(stem_type, stem_res_4_rule_)
                        stem_results.append({"规则":stem_res_4_rule})
                    elif parser_fun == "模型":
                        stem_res_4_model = get_result_4_model(file_re_info, sec_index_re,stem_cn_name,rule_info, context_4_stem_vid, model, tokenizer)
                        stem_results.append({"模型":stem_res_4_model})
                        print(f"模型预测答案为{stem_res_4_model}\n")
                        # pass
                    else:
                        raise print(f"{stem_name}\t{stem_cn_name}\t的解析方式为{parser_fun},错误.")
                else:
                    raise print(f"{vid}就诊中{stem_name}:{stem_cn_name}的stem_info应该解析为dict，但是实际为{stem_info}，错误！")
            # 3.6 观察发现有多条结果的是”禁忌症“相关stem字段，答案只有y，n，合并答案
            stem_res = combine_stem_res_4_rule_and_model(stem_results)
            print(f'{vid}=>,{stem_name}({stem_cn_name})\n{context_4_stem_vid}\n\t{stem_results}===>{stem_res}')
            vid_2_stem_answer[vid][stem_name] = stem_res
            try:
                # 3.7 数据类型为字符串的答案结果应该只有一个，因此需要核对结果
                check_stem_res_and_type(stem_type,stem_res)
            except:
                # print(f"{vid}就诊中{stem_name}:{stem_cn_name}的数据类型为{stem_type}，规则为\n{stem_rule_info}\n预测答案为{stem_res}，不符合要求。\n")
                pass

    # 3.8 根据有向无环图进行后处理
    vid_2_stem_answer = post_processing(vid_2_stem_answer, vid_2_is_dead)
    for vid, stem_answer in vid_2_stem_answer.items():
        for stem, ans in stem_answer.items():
            print(vid, stem, ans)

    # 3.9 保存excel格式的模型预测结果
    vid_2_stem_answer_4_pd = covert_dict_2_pd(vid_2_stem_answer)
    os.makedirs(results_dir_path,exist_ok=True)
    out_fn = "预测结果.xlsx" if is_train else "测试预测结果.xlsx"
    pd.DataFrame(vid_2_stem_answer_4_pd).to_excel(os.path.join(results_dir_path, out_fn), index=False)

    # 4. 将模型预测结果和标注结果对比
    if is_train and os.path.exists(train_gold_annotaion_path):
        compare_results(vid_2_stem_answer, train_gold_annotaion_path)


if __name__ == '__main__':
    is_train = False
    if is_train:
        clinical_data_dir_path = train_clinical_data_dir_path
        prepro_data_dir_path = train_prepro_data_dir_path
        orig_data_dir_path = train_orig_data_dir_path
    else:
        clinical_data_dir_path = test_clinical_data_dir_path
        prepro_data_dir_path =  test_prepro_data_dir_path
        orig_data_dir_path =  test_orig_data_dir_path  
    main()

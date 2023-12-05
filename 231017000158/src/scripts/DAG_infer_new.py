import os
from config.config import *
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# 与各个数据项相关的文档类型
term_2_contexts = {
    "STEMI-1-1-7": [
        "discharge_note",
    ],
    "STEMI-1-1-1": [
        "admit_note",
    ],
    "STEMI-1-2-1": [
        "discharge_note",
    ],
    "STEMI-2-1-1": [
        "medical_order",
    ],
    "STEMI-2-1-3": [
        "medical_order",
    ],
    # STEMI-2-2-1：直接根据就诊流水号是否出现在超声心动图结果来判断
    "STEMI-2-2-3-4": [
        "ultrasonic_results",
    ],
    "STEMI-2-2-3-5": [
        "ultrasonic_results",
    ],
    "STEMI-3-1-3": [
        "discharge_note",
    ],
    "STEMI-3-2-1": [
        "other_notes",
    ],
    "STEMI-3-2-2": [
        "other_notes",
    ],
    "STEMI-3-2-3-1": [
        "other_notes",
    ],
    "STEMI-3-2-3-2-1": [
        "other_notes",
    ],
    "STEMI-3-2-3-2-5": [
        "other_notes",
    ],
    "STEMI-3-2-3-2-2": [
        "other_notes",
    ],
    "STEMI-3-2-3-2-6": [
        "other_notes",
    ],
    "STEMI-3-2-3-2-3": [
        "other_notes",
    ],
    "STEMI-3-2-3-2-7": [
        "other_notes",
    ],
    "STEMI-3-2-3-2-4": [
        "other_notes",
    ],
    "STEMI-3-2-3-2-8": [
        "other_notes",
    ]
}
term_2_contexts2 = {"STEMI-3-2-3-3-2":["pci_note"],
                    "STEMI-3-2-3-3-4":["pci_note"],
                    "STEMI-3-2-3-3-6":["pci_note"],
                    "STEMI-3-2-3-3-8":["pci_note"],
                    "STEMI-3-2-3-3-12":["pci_note"],
                    "STEMI-3-2-7-1":["medical_order"],
                    "STEMI-3-2-7-2":["medical_order"],
                    "STEMI-4-1-1":["medical_order"],
                    "STEMI-4-2":["medical_order"],
                    "STEMI-5-1-1":["medical_order"],
                    "STEMI-5-1-2":["medical_order"],
                    "STEMI-5-2-1":["medical_order"],
                    "STEMI-5-2-2":["medical_order"],
                    "STEMI-5-3-1":["medical_order"],
                    "STEMI-5-3-3":["medical_order"],
                    "STEMI-5-3-4-A":["medical_order"],
                    "STEMI-5-3-4-B":["medical_order"],
                    "STEMI-5-4-1":["medical_order"],
                    "STEMI-5-4-3":["medical_order"],
                    "STEMI-5-4-4":["medical_order"],
                    "STEMI-6-1":["medical_order"],
                    "STEMI-6-1-2":["medical_order"],
                    "STEMI-6-2":["medical_order"],
                    "STEMI-6-2-2":["medical_order"],
                    "STEMI-6-3":["medical_order"],
                    "STEMI-6-3-2":["medical_order"],
                    "STEMI-6-3-2-A":["medical_order"],
                    "STEMI-6-3-2-B":["medical_order"],
                    }
term_2_contexts.update(term_2_contexts2)

# 针对各个数据项的queries
term_2_queries = {
    "STEMI-1-1-7": [
        "你将获得一份病历，请你找出病历中提及心功能分级的内容，并引用原文输出，如果病历中不包含相关内容，则无需输出。\n病历：{}",
        "因此，患者的心功能分级为？\n选项：\nA. 心功能I级\nB. 心功能II级\nC. 心功能III级\nD. 心功能IV级\n请回答A或者B或者C或者D。"
    ],
    "STEMI-1-1-1": [
        "你将获得一份病历，你的任务是根据病历内容判断是否做了心电图检查。\n病历：{}\n是否做了心电图检查？请直接回答“是”或者“否”。",
    ],
    "STEMI-1-2-1": [
        "你将获得一份病历，请你找出病历中出现氯吡格雷、替格瑞洛或者波立维的句子，并引用原文输出，如果病历中不包含相关内容，则无需输出。\n病历：{}",
        "因此，患者是否适合使用P2Y12受体拮抗剂（氯吡格雷、替格瑞洛、波立维等）？请直接回答“是”或者“否”。",
    ],
    "STEMI-2-1-1": [
        "你将获得一份检查报告，你的任务是根据报告内容判断是否做了胸片检查。\n检查报告：\n{}\n是否做了胸片检查？请直接回答“是”或者“否”。",
    ],
    "STEMI-2-1-3": [
        "你将获得一份检查报告，你的任务是根据报告内容判断是否做了胸片检查。\n检查报告：\n{}\n是否做了胸片检查？请直接回答“是”或者“否”。",
        "若做了胸片检查，是否发现肺淤血或肺水肿？请直接回答“是”或者“否”。",
    ],
    # STEMI-2-2-1：直接根据就诊流水号是否出现在超声心动图结果来判断
    "STEMI-2-2-3-4": [
        "你将获得一份超声心动图检查报告，你的任务是根据检查报告的内容回答一个问题。\n超声心动图检查报告：\n{}\n根据检查结论，患者是否确诊左室室壁瘤（注意，节段性室壁运动异常不代表室壁瘤形成）？请直接回答“是”或者“否”。",
    ],
    "STEMI-2-2-3-5": [
        "你将获得一份超声心动图检查报告，你的任务是根据检查报告的内容回答一个问题。\n超声心动图检查报告：\n{}\n根据检查结论，患者是否确诊左心室内血栓？请直接回答“是”或者“否”。",
    ],
    "STEMI-3-1-3": [
        "你将获得一份病历，你的任务是根据病历内容回答一个问题。\n病历：{}\n\n请问病历是否提到了溶栓治疗？如果提到了溶栓治疗，请引用原文输出。",
        "若病历中提到了溶栓治疗，患者是否接受了溶栓治疗？请直接回答“是”或者“否”。"
    ],
    "STEMI-3-2-1": [
        "你将获得一份病历，你的任务是根据病历内容回答一个问题。\n病历：{}\n\n请问病历是否提到了PCI治疗？如果提到了PCI治疗，请引用原文输出。",
        "若病历中提到了PCI治疗，患者是否接受了PCI治疗？请直接回答“是”或者“否”。"
    ],
    "STEMI-3-2-2": [
        "你将获得一份病历，你的任务是根据病历内容回答一个问题。\n病历：{}\n\n请问病历是否提到了PCI治疗？如果提到了PCI治疗，请引用原文输出。",
        "若病历中提到了PCI治疗，患者是否适合PCI治疗？请直接回答“是”或者“否”。"
    ],
    "STEMI-3-2-3-1": [
        # "你将获得一份病历，请你找出病历中提及病变血管的内容，并引用原文输出，如果病历中不包含相关内容，则无需输出。\n病历：{}",
        "你将获得一份病历，请你找出病历中左主干（LM）或三支病变（LAD、LCX、RCA）出现的地方（引用原文输出），如果病历中不包含相关内容，则无需输出。\n病历：{}",
        "根据以上信息，患者的病变血管包括哪些？请从以下选项中选择（多选）：\nA. 左主干\nB. 左前降支（LAD）\nC. 回旋支（LCX）\nD. 右冠状动脉（RCA）"
    ],
    "STEMI-3-2-3-2-1": [
        "你将获得一份病历，请你找出病历中提及左前降支狭窄程度（%）的内容，并引用原文输出，如果病历中不包含相关内容，则无需输出。\n病历：{}",
        "因此，患者的左前降支狭窄程度（%）为？\n选项：\nA. 1-25%\nB. 25-50%\nC. 50-70%\nD. 70-100%\nE. 无法确定\n请回答A或者B或者C或者D或者E。",
    ],
    "STEMI-3-2-3-2-5": [
        "你将获得一份病历，请你找出病历中提及左前降支的TIMI等级的内容，并引用原文输出，如果病历中不包含相关内容，则无需输出。\n病历：{}",
        "因此，患者的左前降支的TIMI等级为？\n选项：\nA. 0级\nB. I级\nC. II级\nD. III级\nE. 无法确定\n请回答A或者B或者C或者D或者E。",
    ],
    "STEMI-3-2-3-2-2": [
        "你将获得一份病历，请你找出病历中提及回旋支狭窄程度（%）的内容，并引用原文输出，如果病历中不包含相关内容，则无需输出。\n病历：{}",
        "因此，患者的回旋支狭窄程度（%）为？\n选项：\nA. 狭窄程度1-25%\nB. 狭窄程度25-50%\nC. 狭窄程度50-70%\nD. 狭窄程度70-100%\nE. 无法确定\n请回答A或者B或者C或者D或者E。",
    ],
    "STEMI-3-2-3-2-6": [
        "你将获得一份病历，请你找出病历中提及回旋支的TIMI等级的内容，并引用原文输出，如果病历中不包含相关内容，则无需输出。\n病历：{}",
        "因此，患者的回旋支的TIMI等级为？\n选项：\nA. 0级\nB. I级\nC. II级\nD. III级\nE. 无法确定\n请回答A或者B或者C或者D或者E。",
    ],
    "STEMI-3-2-3-2-3": [
        "你将获得一份病历，请你找出病历中提及右冠状动脉的狭窄程度（%）的内容，并引用原文输出，如果病历中不包含相关内容，则无需输出。\n病历：{}",
        "因此，患者的右冠状动脉的狭窄程度（%）为？\n选项：\nA. 狭窄程度1-25%\nB. 狭窄程度25-50%\nC. 狭窄程度50-70%\nD. 狭窄程度70-100%\nE. 无法确定\n请回答A或者B或者C或者D或者E。",
    ],
    "STEMI-3-2-3-2-7": [
        "你将获得一份病历，请你找出病历中提及右冠状动脉的TIMI等级的内容，并引用原文输出，如果病历中不包含相关内容，则无需输出。\n病历：{}",
        "因此，患者的右冠状动脉的TIMI等级为？\n选项：\nA. 0级\nB. I级\nC. II级\nD. III级\nE. 无法确定\n请回答A或者B或者C或者D或者E。",
    ],
    "STEMI-3-2-3-2-4": [
        "你将获得一份病历，请你找出病历中提及左主干狭窄程度（%）的内容，并引用原文输出，如果病历中不包含相关内容，则无需输出。\n病历：{}",
        "因此，患者的左主干狭窄程度（%）为？\n选项：\nA. 狭窄程度1-25%\nB. 狭窄程度25-50%\nC. 狭窄程度50-70%\nD. 狭窄程度70-100%\nE. 无法确定\n请回答A或者B或者C或者D或者E。",
    ],
    "STEMI-3-2-3-2-8": [
        "你将获得一份病历，请你找出病历中提及左主干的TIMI等级的内容，并引用原文输出，如果病历中不包含相关内容，则无需输出。\n病历：{}",
        "因此，患者的左主干的TIMI等级为？\n选项：\nA. 0级\nB. I级\nC. II级\nD. III级\nE. 无法确定\n请回答A或者B或者C或者D或者E。",
    ],
}
term_2_queries2 = {}
term_2_queries.update(term_2_queries2)

# 各数据项所依赖的数据项
term_dependencies = {
    "STEMI-1-1-7": [],
    "STEMI-1-1-1": [],
    "STEMI-1-2-1": [],
    "STEMI-2-1-1": [],
    "STEMI-2-1-3": ["STEMI-2-1-1"],
    "STEMI-2-2-1": [],
    "STEMI-2-2-3-4": ["STEMI-2-2-1"],
    "STEMI-2-2-3-5": ["STEMI-2-2-1"],
    "STEMI-3-1-3": [],
    "STEMI-3-2-1": [],
    "STEMI-3-2-2": ["STEMI-3-2-1"],
    "STEMI-3-2-3-1": ["STEMI-3-2-1", "STEMI-3-2-2"],
    "STEMI-3-2-3-2-1": ["STEMI-3-2-3-1"],
    "STEMI-3-2-3-2-5": ["STEMI-3-2-3-1"],
    "STEMI-3-2-3-2-2": ["STEMI-3-2-3-1"],
    "STEMI-3-2-3-2-6": ["STEMI-3-2-3-1"],
    "STEMI-3-2-3-2-3": ["STEMI-3-2-3-1"],
    "STEMI-3-2-3-2-7": ["STEMI-3-2-3-1"],
    "STEMI-3-2-3-2-4": ["STEMI-3-2-3-1"],
    "STEMI-3-2-3-2-8": ["STEMI-3-2-3-1"],
    "STEMI-3-2-3-3-2": ["STEMI-3-2-3-1"],
    "STEMI-3-2-3-3-4": ["STEMI-3-2-3-1"],
    "STEMI-3-2-3-3-6": ["STEMI-3-2-3-1"],
    "STEMI-3-2-3-3-8": ["STEMI-3-2-3-1"],
    "STEMI-3-2-3-3-12": ["STEMI-3-2-3-1"],
    "STEMI-3-2-7-1": ["STEMI-3-2-1"],
    "STEMI-3-2-7-2": ["STEMI-3-2-7-1"],
    "STEMI-4-1-1": [],
    "STEMI-4-2": ["STEMI-4-1-1"],
    "STEMI-5-1-1": [],
    "STEMI-5-1-2": ["STEMI-5-1-1"],
    "STEMI-5-2-1": ["STEMI-4-1-1"],
    "STEMI-5-2-2": ["STEMI-5-2-1"],
    "STEMI-5-3-1": [],
    "STEMI-5-3-3": ["STEMI-5-3-1"],
    "STEMI-5-3-4-A": ["STEMI-5-3-3"],
    "STEMI-5-3-4-B": ["STEMI-5-3-3"],
    "STEMI-5-4-1": [],
    "STEMI-5-4-3": ["STEMI-5-4-1"],
    "STEMI-5-4-4": ["STEMI-5-4-3"],
    "STEMI-6-1": [],
    "STEMI-6-1-2": ["STEMI-6-1"],
    "STEMI-6-2": ["STEMI-4-1-1"],
    "STEMI-6-2-2": ["STEMI-6-2"],
    "STEMI-6-3": ["STEMI-5-3-1"],
    "STEMI-6-3-2": ["STEMI-5-3-1", "STEMI-6-3"],
    "STEMI-6-3-2-A": ["STEMI-5-3-1", "STEMI-6-3-2"],
    "STEMI-6-3-2-B": ["STEMI-5-3-1", "STEMI-6-3-2"],
}

# 取值等于对应值时必填
must_answer = {
    ("STEMI-2-1-3", "STEMI-2-1-1"): "y",
    ("STEMI-2-2-3-4", "STEMI-2-2-1"): "y",
    ("STEMI-2-2-3-5", "STEMI-2-2-1"): "y",
    ("STEMI-3-2-2", "STEMI-3-2-1"): "y",
    ("STEMI-3-2-3-1", "STEMI-3-2-1"): "y",
    ("STEMI-3-2-3-1", "STEMI-3-2-2"): "n",
    ("STEMI-3-2-3-2-1","STEMI-3-2-3-1"): "a",
    ("STEMI-3-2-3-2-5","STEMI-3-2-3-1"): "a",
    ("STEMI-3-2-3-2-2","STEMI-3-2-3-1"): "b",
    ("STEMI-3-2-3-2-6","STEMI-3-2-3-1"): "b",
    ("STEMI-3-2-3-2-3","STEMI-3-2-3-1"): "c",
    ("STEMI-3-2-3-2-7","STEMI-3-2-3-1"): "c",
    ("STEMI-3-2-3-2-4","STEMI-3-2-3-1"): "d",
    ("STEMI-3-2-3-2-8","STEMI-3-2-3-1"): "d",
    ("STEMI-3-2-7-1", "STEMI-3-2-1"): "y",
    ("STEMI-3-2-7-2", "STEMI-3-2-7-1"): "y",
    ("STEMI-4-2", "STEMI-4-1-1"): "n",
    ("STEMI-5-1-2", "STEMI-5-1-1"): "y",
    ("STEMI-5-2-1", "STEMI-4-1-1"): "n",
    ("STEMI-5-2-2", "STEMI-5-2-1"): "y",
    ("STEMI-5-3-3", "STEMI-5-3-1"): "n",
    ("STEMI-5-3-4-A", "STEMI-5-3-3"): "y",
    ("STEMI-5-3-4-B", "STEMI-5-3-3"): "y",
    ("STEMI-5-4-3", "STEMI-5-4-1"): "n",
    ("STEMI-5-4-4", "STEMI-5-4-3"): "y",
    ("STEMI-6-1-2", "STEMI-6-1"): "y",
    ("STEMI-6-2", "STEMI-4-1-1"): "n",
    ("STEMI-6-2-2", "STEMI-6-2"): "y",
    ("STEMI-6-3-2", "STEMI-6-3"): "y",
    ("STEMI-6-3-2-A", "STEMI-6-3-2"): "a",
    ("STEMI-6-3-2-B", "STEMI-6-3-2"): "b",
}
# 可填写的值
possable_answer={
    ("STEMI-3-2-7-1","STEMI-3-2-1"): "y",
}
# 取值等于对应值时无需填写
need_not_answer = {
    ("STEMI-3-2-3-2-1","STEMI-3-2-3-1"): "UTD",
    ("STEMI-3-2-3-2-5","STEMI-3-2-3-1"): "UTD",
    ("STEMI-3-2-3-2-2","STEMI-3-2-3-1"): "UTD",
    ("STEMI-3-2-3-2-6","STEMI-3-2-3-1"): "UTD",
    ("STEMI-3-2-3-2-3","STEMI-3-2-3-1"): "UTD",
    ("STEMI-3-2-3-2-7","STEMI-3-2-3-1"): "UTD",
    ("STEMI-3-2-3-2-4","STEMI-3-2-3-1"): "UTD",
    ("STEMI-3-2-3-2-8","STEMI-3-2-3-1"): "UTD",
    ("STEMI-3-2-3-3-2","STEMI-3-2-3-1"): "UTD",
    ("STEMI-3-2-3-3-4","STEMI-3-2-3-1"): "UTD",
    ("STEMI-3-2-3-3-6","STEMI-3-2-3-1"): "UTD",
    ("STEMI-3-2-3-3-8","STEMI-3-2-3-1"): "UTD",
    ("STEMI-3-2-3-3-12","STEMI-3-2-3-1"): "UTD",
    ("STEMI-6-3", "STEMI-5-3-1"): "y",
    ("STEMI-6-3-2", "STEMI-5-3-1"): "y",
    ("STEMI-6-3-2-A", "STEMI-5-3-1"): "y",
    ("STEMI-6-3-2-A", "STEMI-5-3-1"): "y",
    ("STEMI-6-3-2-B", "STEMI-5-3-1"): "y",
    ("STEMI-3-2-3-3-2", "STEMI-3-2-3-1"): "UTD",
    ("STEMI-3-2-3-3-4", "STEMI-3-2-3-1"): "UTD",
    ("STEMI-3-2-3-3-6", "STEMI-3-2-3-1"): "UTD",
    ("STEMI-3-2-3-3-8", "STEMI-3-2-3-1"): "UTD",
    ("STEMI-3-2-3-3-12", "STEMI-3-2-3-1"): "UTD",
}

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

    q = set()
    for term, in_deg in degs.items():
        if in_deg == 0:
            q.add(term)

    while len(q) > 0:
        front = q.pop()
        res.append(front)

        for term in tails[front]:
            degs[term] -= 1
            if degs[term] == 0:
                q.add(term)

    return res


# 对模型回复进行解析得到答案
def parse_answer(response, term):
    answer = None
    # TODO
    return answer
def get_inspection_results(patient_id_2_info):
    path_other_results = os.path.join(orig_data_dir_path,"6-其他检查结果.xlsx")
    other_results = pd.read_excel(path_other_results)
    num_lines = len(other_results)
    for i in range(num_lines):
        data_dict = dict(other_results.loc[i])
        check_item = data_dict['检查项目']
        check_observation = data_dict['检查所见']
        check_conclusion = data_dict['检查结论']
        content = '检查项目：' + check_item + '\t检查所见：' + check_observation + '\t检查结论：' + check_conclusion + '\n'

        patient_id = str(data_dict['就诊流水号'])
        patient_id_2_info[patient_id]['other_results'] += content
    for patient_id, info in patient_id_2_info.items():
        if info['other_results'] == "":
            info['other_results'] = "未进行胸片检查或其他检查。\n"


def get_ultrasonic_results(patient_id_2_info):
    path_ultrasonic_results = os.path.join(orig_data_dir_path,"5-超声心动图结果.xlsx")
    ultrasonic_results = pd.read_excel(path_ultrasonic_results)
    num_lines = len(ultrasonic_results)
    for i in range(num_lines):
        data_dict = dict(ultrasonic_results.loc[i])
        check_observation = data_dict['检查所见']
        check_conclusion = data_dict['检查结论']
        content = '检查所见：\n' + check_observation + '\n检查结论：\n' + check_conclusion + '\n\n'

        patient_id = str(data_dict['就诊流水号'])
        patient_id_2_info[patient_id]['ultrasonic_results'] += content
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("/home/u21s051047/huggingface/chatglm2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("/home/u21s051047/huggingface/chatglm2-6b", trust_remote_code=True, device='cuda:0')
    '''
    """ 初始化每个就诊流水号对应的就诊信息 patient_id_2_info """
    patient_id_2_info = {}  # pid_vid_table_name_2_info
    list_patient_ids = []

    fname_patient_ids = '健康医疗大数据主题赛/赛题6-初赛材料_最终版/3-就诊流水号列表.txt'
    f = open(fname_patient_ids, encoding='utf-8')
    for line in f.readlines():
        patient_id = line.strip()

        assert patient_id not in patient_id_2_info

        info = {
            "admit_note": "",           # 入院记录
            "discharge_note": "",       # 出院记录
            "other_notes": [],          # 其他病历文书（如：冠状动脉造影及PCI记录）
            "ultrasonic_results": "",   # 超声心动图结果
            "other_results": "",        # 其他检查结果
            "medical_order": "",        # 医嘱
        }

        patient_id_2_info[patient_id] = info
        list_patient_ids.append(patient_id)
    list_patient_ids = sorted(list_patient_ids)

    """ 读取文档材料，填入 patient_id_2_info """
    # dir_discharge_notes = '健康医疗大数据主题赛/赛题6-初赛材料_最终版/4-病历文书'
    dir_discharge_notes = clinical_data_dir_path
    for fn in os.listdir(dir_discharge_notes):
        assert fn[-3:] == 'txt'
        fpath = os.path.join(dir_discharge_notes, fn)
        f = open(fpath, encoding='utf-8')
        text = f.read()

        doc_type = None # todo
        for name_cn, name_en in doc_type_dict.items():
            if name_cn in fn:
                doc_type = name_en
                break
        if not doc_type:
            doc_type = "other_notes"


        for patient_id, info in patient_id_2_info.items():
            if patient_id in fn:
                if isinstance(info[doc_type], list):
                    info[doc_type].append(text)
                else:
                    assert info[doc_type] == ""
                    info[doc_type] = text
                break


    get_inspection_results(patient_id_2_info)
    get_ultrasonic_results(patient_id_2_info)


    """ 以上内容和infer.py相同，以下为变更内容 """
    '''
    # 有向无环图
    degs, tails = prepare_degs_and_tails(term_dependencies)
    # 拓扑排序
    term_seq = toposort(degs, tails)
    assert len(term_seq) == len(term_dependencies)

    # 对于每个数据项，各个就诊流水号对应的预测答案
    term_2_patient_id_2_answer = dict((term, {}) for term in term_seq)

    # 按依赖次序，根据各数据项相关的文档以问答形式预测数据项的值
    for term in term_seq:
        queries = term_2_queries[term]
        contexts = term_2_contexts[term]
        assert len(contexts) == 1
        context = contexts[0]

        # 该数据项所依赖的所有数据项
        dependent_terms = term_dependencies[term]

        # 对于各个就诊患者
        list_patient_ids=[] # todo
        for patient_id in list_patient_ids:
            # 根据所依赖数据项的预测值判断是否需要填写当前数据项
            flag_must_answer = False
            flag_need_not_answer = False
            for dependent_term in dependent_terms:
                dependent_term_answer = term_2_patient_id_2_answer[dependent_term][patient_id]

                if (term, dependent_term) in must_answer and dependent_term_answer == must_answer((term, dependent_term)):
                    flag_must_answer = True

                if (term, dependent_term) in need_not_answer and dependent_term_answer == need_not_answer((term, dependent_term)):
                    flag_need_not_answer = True

            if flag_must_answer:
                assert flag_need_not_answer == False

            # 不需要填写当前数据项
            if flag_need_not_answer:
                term_2_patient_id_2_answer[term][patient_id] = None
                continue

            """ 预测当前数据项的值 """
            patient_id_2_info = {} # todo {vid:info}
            info = patient_id_2_info[patient_id]
            context_content = info[context]
            if isinstance(context_content, str):
                context_content = [context_content]

            print("-"*50, patient_id, term)
            # 对每一个相关文档内容
            for content in context_content:
                # 多轮问答
                h = []
                for query in queries:
                    try:
                        query = query.format(content)
                    except:
                        pass

                    print(query)
                    print()

                    r, h = model.chat(tokenizer, query, history=h, do_sample=False)
                    print(r)
                    print()

            print("-"*50, patient_id, term)
            print(flush=True)

            # 从模型回复中解析出答案
            answer = parse_answer(r, term)
            term_2_patient_id_2_answer[term][patient_id] = answer
            
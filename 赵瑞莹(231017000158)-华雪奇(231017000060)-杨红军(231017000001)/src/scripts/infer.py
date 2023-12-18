import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("/home/u21s051047/huggingface/chatglm2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("/home/u21s051047/huggingface/chatglm2-6b", trust_remote_code=True, device='cuda:1')

    # 与各个数据项相关的文档类型
    term_2_contexts = {
        "STEMI-1-1-7": [
            "discarge_note",
        ],
        "STEMI-1-1-1": [
            "admit_note",
        ],
        "STEMI-1-2-1": [
            "discarge_note",
        ],
        "STEMI-2-1-1": [
            "other_results",
        ],
        "STEMI-2-1-3": [
            "other_results",
        ],
        # STEMI-2-2-1：直接根据就诊流水号是否出现在超声心动图结果来判断
        "STEMI-2-2-3-4": [
            "ultrasonic_results",
        ],
        "STEMI-2-2-3-5": [
            "ultrasonic_results",
        ],
        "STEMI-3-1-3": [
            "discarge_note",
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

    """ 初始化每个就诊流水号对应的就诊信息 patient_id_2_info """
    patient_id_2_info = {}
    list_patient_ids = []

    fname_patient_ids = '健康医疗大数据主题赛/赛题6-初赛材料_最终版/3-就诊流水号列表.txt'
    f = open(fname_patient_ids, encoding='utf-8')
    for line in f.readlines():
        patient_id = line.strip()

        assert patient_id not in patient_id_2_info

        info = {
            "admit_note": "",           # 入院记录
            "discarge_note": "",        # 出院记录
            "other_notes": [],          # 其他病历文书（如：冠状动脉造影及PCI记录）
            "ultrasonic_results": "",   # 超声心动图结果
            "other_results": "",        # 其他检查结果
            "medical_order": "",        # 医嘱
        }

        patient_id_2_info[patient_id] = info
        list_patient_ids.append(patient_id)
    list_patient_ids = sorted(list_patient_ids)

    """ 读取文档材料，填入 patient_id_2_info """
    dir_discarge_notes = '健康医疗大数据主题赛/赛题6-初赛材料_最终版/4-病历文书'
    for fn in os.listdir(dir_discarge_notes):
        assert fn[-3:] == 'txt'
        fpath = os.path.join(dir_discarge_notes, fn)
        f = open(fpath, encoding='utf-8')
        text = f.read()

        doc_type = None
        if "入院记录" in fn:
            doc_type = "admit_note"
        elif "出院记录" in fn:
            doc_type = "discarge_note"
        else:
            doc_type = "other_notes"

        for patient_id, info in patient_id_2_info.items():
            if patient_id in fn:
                if isinstance(info[doc_type], list):
                    info[doc_type].append(text)
                else:
                    assert info[doc_type] == ""
                    info[doc_type] = text
                break

    path_other_results = '健康医疗大数据主题赛/赛题6-初赛材料_最终版/6-其他检查结果.xlsx'
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

    path_ultrasonic_results = '健康医疗大数据主题赛/赛题6-初赛材料_最终版/5-超声心动图结果.xlsx'
    ultrasonic_results = pd.read_excel(path_ultrasonic_results)
    num_lines = len(ultrasonic_results)
    for i in range(num_lines):
        data_dict = dict(ultrasonic_results.loc[i])
        check_observation = data_dict['检查所见']
        check_conclusion = data_dict['检查结论']
        content = '检查所见：\n' + check_observation + '\n检查结论：\n' + check_conclusion + '\n\n'

        patient_id = str(data_dict['就诊流水号'])
        patient_id_2_info[patient_id]['ultrasonic_results'] += content


    """ 根据各数据项相关的文档以问答形式预测数据项的值 """
    for term, queries in term_2_queries.items():
        contexts = term_2_contexts[term]

        assert len(contexts) == 1
        context = contexts[0]

        # 对于各个就诊患者
        for patient_id in list_patient_ids:
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

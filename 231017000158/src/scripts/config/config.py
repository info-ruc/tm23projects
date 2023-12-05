import os.path
import re

# 1. 文件
config_dir_path = os.path.abspath(os.path.dirname(__file__))
program_dir_path = os.path.join(config_dir_path, os.path.pardir)
# 以下是data目录
data_dir_path = os.path.join(program_dir_path, 'data')
orig_data_dir_path = os.path.join(data_dir_path, 'orig_datas')
clinical_data_dir_path = os.path.join(orig_data_dir_path, '4-病历文书')
stem_file = "STEMI数据项说明_备注说明_补充信息_v4.xlsx"
prepro_data_dir_path = os.path.join(data_dir_path, 'prepro_data')
prepro_orig_data_dir_path = os.path.join(data_dir_path, 'prepro_orig_datas')
results_dir_path = os.path.join(program_dir_path, 'results')
# 以下是model目录
model_dir_path = os.path.join(program_dir_path,"model")

# 2. doctype
doc_type_dict={"入院记录":"admit_note",
               "出院记录":"discharge_note",
               "冠状动脉造影":"pci_note", # 急诊冠状动脉造影及PCI记录,择期冠状动脉造影记录,择期冠状动脉造影及PCI记录
                "死亡记录":"death_note",
               "手术记录":"operate_note",
               "永久起搏器植入术记录": "pacemaker_note",
               "植入心脏起搏器手术报告": "pacemaker_heart_note",
               "超声心动图结果":"ultrasound_result",
               "医嘱":"medical_order",
               "其他记录":"other_notes",
                "检验报告":"lab_report"}
# second_table_dir_dict 根据正则获取核心词，然后根据核心词获取二级表名
# 中文的正则表达式是
sc_table_dict_re = {"admit_note":"^\S\s\S\s\S\s\S|^[\u4e00-\u9fa5 ]*?[：:]\s*$",
                     "discharge_note":"^[一二三四五六七八九十]*、[\u4e00-\u9fa5 ]+?[:：]",
                    "death_note": "^[一二三四五六七八九十]*、[\u4e00-\u9fa5 ]+?[:：]",
                    "pci_note": "(择期冠状动脉造影及PCI记录|急诊冠状动脉造影及PCI记录|择期冠状动脉造影记录|结论|备注|介入治疗结果|介入治疗基本信息|介入时间|术后医嘱|并发症|辅助设备|其他影像学|术后安返病房)"}
# 无意义词列表 正则re
stop_words = ['的', '是', '在', '我', '你',"否","有","无","\*+|[a-z][: ]|def|请|是否|有无|[\n\s\t\\t\\n]+|[、，。；\?]+","使用","的","选择",":",";"]

# 默认值n

# 备注3中的信息进行解析
# rule_2_parser = "(?P<first_layer>.*?)\.(?P<sec_layer>.*?)【(?P<re_model>.*?)】[:：](?P<info>.*)"
# rule_2_ls_parser = "[;；]"

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
# 取值等于对应值时可填写
may_answer={
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
}


infer_answer_via_dependency = {
    "STEMI-3-2-2": {
        "STEMI-3-2-1": {
            "y": "n",
        }
    }
}


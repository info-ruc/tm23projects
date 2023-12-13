import os
import re
import pandas as pd
import time

# 从入院记录获取入院时间
def get_admit_time(admit_note=None, admit_note_fpath=None, prefix="入院时间：	"):
    assert admit_note is not None or admit_note_fpath is not None

    if admit_note is None:
        admit_note_f = open(admit_note_fpath, encoding='utf-8')
        admit_note = admit_note_f.read()

    time_string = None
    for line in admit_note.split('\n'):
        if prefix in line:
            time_string = re.search(prefix+r"(.*)", line).group()[len(prefix):].strip()
            # print(time_string)
            break

    assert time_string is not None
    adimit_time = time.strptime(time_string, "%Y年%m月%d日 %H:%M")

    return adimit_time

# 计算两个时间的时间间隔（秒）
def cal_time_interval(time_struct1, time_struct2):
    return abs(time.mktime(time_struct1) - time.mktime(time_struct2))


if __name__ == '__main__':
    
    patient_id_2_admit_time = {}

    """ 读取文档材料，得到 patient_id_2_admit_time """
    dir_discarge_notes = '健康医疗大数据主题赛/赛题6-初赛材料_最终版/4-病历文书'
    for fn in os.listdir(dir_discarge_notes):
        assert fn[-3:] == 'txt'
        fpath = os.path.join(dir_discarge_notes, fn)
        f = open(fpath, encoding='utf-8')
        text = f.read()

        if "入院记录" in fn:
            patient_id = fn.split("_")[0]
            admit_time = get_admit_time(text)
            patient_id_2_admit_time[patient_id] = admit_time

    """ 获取每个医嘱的医嘱开始时间，并计算其与入院时间的时间间隔 """
    path_medical_order = '健康医疗大数据主题赛/赛题6-初赛材料_最终版/7-医嘱.xlsx'
    medical_orders = pd.read_excel(path_medical_order)
    num_lines = len(medical_orders)
    for i in range(num_lines):
        data_dict = dict(medical_orders.loc[i])
        patient_id = str(data_dict['就诊流水号'])
        order_time = str(data_dict['医嘱开始时间'])
        assert isinstance(order_time, str)
        order_time = time.strptime(order_time, "%Y%m%d%H%M%S")
        # print(order_time)
        
        admit_time = patient_id_2_admit_time[patient_id]
        time_interval = cal_time_interval(order_time, admit_time)
        # print(time_interval)

        # 一个小时3600秒，一天24小时
        if time_interval < 3600 * 24:
            # 在同一天
            print(patient_id_2_admit_time[patient_id], order_time)
        else:
            print("不在同一天")
            
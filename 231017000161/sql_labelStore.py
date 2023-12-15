import pandas as pd
import pymysql

import sql_utils

db = pymysql.connect(host="localhost",
                     user="fin",
                     password="guorp",
                     db="financial",
                     port=3306,
                     charset='utf8mb4')
cur = db.cursor()


def clearMysql():
    sql_utils.clear_table('resume_label_store')


def insert(data):
    sql_utils.clear_table('resume_label_store')
    col_names = 'field,lclass,label,rule,important,alias_label,needadj'
    sql_utils.insert_table_multi('resume_label_store', data, col_names)


def getAllLabelStore_bak():
    results = sql_utils.query_table('resume_label_store', 'id,lclass,label,rule,important,alias_label,needadj',
                                    ' 1=1 order by alias_label,id')
    label_dict = {}
    label_alias_list = []

    label_tmp = ''
    for result in results:
        if (result[5] != label_tmp):
            if (label_tmp != ''):
                label_dict[label_tmp] = label_alias_list
            label_alias_list = []
            label_alias_list.append(list(result))
            label_tmp = result[5]
        else:
            label_alias_list.append(list(result))
    label_dict[label_tmp] = label_alias_list
    return label_dict


def getAllLabelStore():
    results = sql_utils.query_table('resume_label_store',
                                    'id,lclass,label,rule,important,alias_label,needadj')
    label_dict = {}
    for result in results:
        # print(result[5])
        labels = eval(result[5])
        for label_tmp in labels:
            if label_tmp not in label_dict:
                label_dict[label_tmp] = []
            label_dict[label_tmp].append(list(result))
    return label_dict


def get_label(path):
    label_list = []
    dataframe = pd.read_csv(path, low_memory=False, encoding="utf-8")

    for index, line in dataframe.iterrows():
        label_info = {}
        label_info['field'] = line['field']
        label_info['lclass'] = line['class']
        label_info['label'] = line['label']
        label_info['rule'] = line['rule']
        label_info['important'] = line['important']
        label_info['alias_label'] = line['alias']
        label_info['needadj'] = line['needadj']
        label_list.append(label_info)
    return label_list


def insert_labelStore():
    label_list = get_label('./source/label.csv')
    clearMysql()
    insert(label_list)


if __name__ == '__main__':
    label_list = get_label('./source/label.csv')
    clearMysql()
    insert(label_list)


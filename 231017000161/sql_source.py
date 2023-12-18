import pandas as pd
import pymysql

import sql_utils
from sql_utils import clear_table, insert_table_multi
from utils_print import print_cross_platform, print_skip

database = pymysql.connect(host="localhost", user="fin",
                           password="guorp", db="financial",
                           port=3306, charset='utf8mb4')
cur = database.cursor()
table = 'resume_source'


def push_data(path):
    clear_table(table)
    print_cross_platform("clear_table done")

    useful_infos = [
        ('ADV', ["ADV"]),
        ('COMPETENCY_ADV', ["COMPETENCY_ADV"]),
        ('POSITION', ["POSITION"]),
        ('SCHOLARSHIP', ["SCHOLARSHIP"]),
        ('OTHER_AWARD', ["OTHER_AWARD"]),
        ('COMPUTER_ABILITY', ["COMPUTER_ABILITY"]),
        ('CA', ["COMPETENCY_ADV", "ADV"]),
        ('CC', ["COMPETENCY_ADV", "COMPUTER_ABILITY"]),
        ('CO', ["COMPETENCY_ADV", "OTHER_AWARD"]),
    ]
    if 'temp' in table:
        info_experience = pd.read_csv(path + "RESULT(info,experience).csv", low_memory=False, encoding="utf-8")
    else:
        info_experience = pd.read_csv(path + "small_push.csv", low_memory=False, encoding="utf-8")
    data = []
    print_cross_platform("Load done")
    for index, line in info_experience.iterrows():
        resumeid = str(line['RESUME_ID'])
        if (resumeid != 'nan'):
            resumeid = int(float(resumeid))
            for colname, infos in useful_infos:
                content = ""
                for info in infos:
                    if (str(line[info]) != 'nan'):
                        content += str(line[info]) + r'。'
                if (content != ""):
                    norm_content = content.lower().replace('<br>', '。').replace('\n', '。') \
                        .replace(' ', '，') \
                        .replace('（', '').replace('）', '').replace('“', '').replace('”', '') \
                        .replace('(', '').replace(')', '').replace('‘', '').replace('’', '') \
                        .replace('"', '').replace("'", '').replace('《', '').replace('》', '')
                    # print_cross_platform(content)
                    dataone = {'resumeid': resumeid, 'colname': colname,
                               'content': content, 'norm_content': norm_content
                               }
                    data.append(dataone)
        print_skip("data line", sep='\t', interval=(1 << 15) - 1, print_item='num')
    insert_table_multi(table, data, "resumeid,colname,content,norm_content")
    database.commit()


def getSourceById(id, num, col_names='resumeid,colname,norm_content', other_cond=""):
    sql_text = 'select max(tmp.resumeid) as cnum from (' \
               'select DISTINCT resumeid from resume_source ' \
               'where resumeid>{} {} order by resumeid limit {}) as tmp' \
               ''.format(id, other_cond, num)
    results = sql_utils.query_by_sql(sql_text)
    maxid = results[0][0]
    if maxid is None:
        return None, None
    results = sql_utils.query_table('resume_source', col_names,
                                    'resumeid>{} and resumeid<={} {} order by resumeid,id'
                                    ''.format(id, maxid, other_cond))

    source_dict = {}
    source_id_list = []
    resumeid_tmp = ''
    for result in results:
        if (result[0] != resumeid_tmp):
            if (resumeid_tmp != ''):
                source_dict[resumeid_tmp] = source_id_list
            source_id_list = []
            source_id_list.append(list(result))
            resumeid_tmp = result[0]
        else:
            source_id_list.append(list(result))
    source_dict[resumeid_tmp] = source_id_list
    return source_dict, source_id_list[0][0]


def getSourceAndSimById(id, num, col_names='resumeid,colname,norm_content', other_cond=""):
    sql_text = 'select max(tmp.resumeid) as cnum from (' \
               'select DISTINCT resumeid from resume_source ' \
               'where resumeid>{} {} order by resumeid limit {}) as tmp' \
               ''.format(id, other_cond, num)
    results = sql_utils.query_by_sql(sql_text)
    maxid = results[0][0]
    if maxid is None:
        return None, None, None
    results = sql_utils.query_table('resume_source', col_names,
                                    'resumeid>{} and resumeid<={} {} order by resumeid,id'
                                    ''.format(id, maxid, other_cond))
    sim = sql_utils.query_table('resume_docsim', 'resumeid,similarity',
                                'resumeid>{} and resumeid<={} {} order by resumeid'
                                ''.format(id, maxid, other_cond))

    source_dict = {}
    source_id_list = []
    # 转格式，汇集同样resume_id的各条数据
    resumeid_tmp = ''
    for result in results:
        if (result[0] != resumeid_tmp):
            if (resumeid_tmp != ''):
                source_dict[resumeid_tmp] = source_id_list
            source_id_list = []
            source_id_list.append(list(result))
            resumeid_tmp = result[0]
        else:
            source_id_list.append(list(result))
    source_dict[resumeid_tmp] = source_id_list
    return source_dict, source_id_list[0][0], sim


if __name__ == '__main__':
    data_path = r"D:\Data" + "\\"


    # push_data(data_path)

    # print(getSourceById(1, 2))

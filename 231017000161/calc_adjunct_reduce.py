import pandas as pd

import sql_adjunct
import sql_adjunct_beta
import sql_adjunct_reduce

BATCH_NUMBER = 300


def getAllAdjunctStore(path):
    adjunct_dict = {}
    dataframe = pd.read_csv(path, low_memory=False, encoding="utf-8")

    for index, line in dataframe.iterrows():
        if line['word'] in adjunct_dict:
            adjunct_dict[line['word']].append({'adjunct': line['adjunct'], 'level': line['level']})
        else:
            adjunct_dict[line['word']] = [{'adjunct': line['adjunct'], 'level': line['level']}]

    return adjunct_dict


# 获取标签库
adjunctStore = getAllAdjunctStore('./source/adjunct.csv')


def calc_adjunct_reduce(taskid):
    sql_adjunct_reduce.clear(taskid)
    start_i = -1
    i = 1
    while True:
        print('{} batch: startid={}'.format(i, start_i))
        i += 1
        (datas, start_i) = sql_adjunct.queryBatch(start_i, BATCH_NUMBER,
                                                  where_clause=' and taskid={} group by resumeid,colname,labelid '
                                                               'order by resumeid,colname,labelid'
                                                               ''.format(taskid))
        if datas is None:
            break
        sql_adjunct_reduce.insert(datas)


def calc_adjunct_reduce_score(taskid):
    start_i = -1
    i = 1
    while True:
        print('{} batch: startid={}'.format(i, start_i))
        i += 1
        (datas, start_i) = sql_adjunct_beta.queryBatch(start_i, BATCH_NUMBER,
                                                       other_cond='and taskid={}'.format(taskid))
        if datas is None:
            break
        # 'resumeid,colname,labelid,label,word,adjunct,important'
        score_dict = {}
        for data in datas:

            conditiontmp = " resumeid='" + str(data[0]) + "' and labelid='" + str(data[2]) + "'"

            if data[4] in adjunctStore:
                adjunctOfLabelList = adjunctStore[data[4]]
            else:
                score_dict[conditiontmp] = 2 * data[6]
                continue

            for adjunctOfLabel in adjunctOfLabelList:
                if adjunctOfLabel['adjunct'] in data[5]:
                    if str(adjunctOfLabel['level']) == 'nan' or str(adjunctOfLabel['level']) == '':
                        adjunctOfLabel['level'] = 2
                    if conditiontmp in score_dict:
                        score_dict[conditiontmp] += adjunctOfLabel['level'] * data[6]
                    else:
                        score_dict[conditiontmp] = adjunctOfLabel['level'] * data[6]
                    break
            if conditiontmp not in score_dict:
                score_dict[conditiontmp] = 2 * data[6]
        sql_adjunct_reduce.update(score_dict, taskid)
        # sql_adjunct_reduce.uniform()


if __name__ == '__main__':
    # calc_adjunct_reduce(-1)
    calc_adjunct_reduce_score(-1)
    # sql_adjunct_reduce.uniform()

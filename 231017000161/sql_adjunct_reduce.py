import sql_utils


def insert(data):
    col_names = 'resumeid,colname,labelid,reducenum,taskid'
    sql_utils.insert_table_multi_bylist('resume_adjunct_reduce', data, col_names)


def update(datas, taskid):
    for key in datas:
        sql_utils.update_table('resume_adjunct_reduce',
                               {'scores': datas[key], 'taskid': taskid},
                               'scores,taskid', key, False)
    sql_utils.force_commit()


def uniform():
    max = sql_utils.query_by_sql('select max(scores) from resume_adjunct_reduce')
    maxscore = 1
    if max[0][0] != 0:
        maxscore = max[0][0]

    sql_utils.update_table_statement('resume_adjunct_reduce', ' scores = ROUND(scores*100/' + str(maxscore) + ',2)',
                                     ' 1=1 ')


def queryBatch(resumeid, num, other_cond=""):
    sql_text = 'select max(tmp.resumeid) as cnum from (' \
               'select DISTINCT resumeid from resume_adjunct_reduce ' \
               'where resumeid>{} {} order by resumeid limit {}) as tmp' \
               ''.format(resumeid, other_cond, num)
    results = sql_utils.query_by_sql(sql_text)
    maxid = results[0][0]
    if maxid is None:
        return None, None
    results = sql_utils.query_by_sql(
        'select t1.resumeid,t2.field,sum(t1.scores) '
        'from resume_adjunct_reduce t1 inner join resume_label_store t2 '
        'where t1.labelid = t2.id and t1.resumeid>{} and t1.resumeid<={} {} '
        'group by t1.resumeid,t2.field order by t1.resumeid,t2.field'
        ''.format(resumeid, maxid, other_cond))
    if len(results) == 0:
        return None, None
    return results, results[-1][0]


def clear(taskid):
    sql_utils.clear_table_condition('resume_adjunct_reduce', "taskid={}".format(taskid))


def close():
    sql_utils.close()

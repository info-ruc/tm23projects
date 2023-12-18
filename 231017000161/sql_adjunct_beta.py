import sql_utils


def insert(data):
    col_names = 'resumeid,colname,labelid,word,adjunct,times,taskid'
    sql_utils.insert_table_multi_bylist('resume_adjunct_beta', data, col_names)


def queryBatch(resumeid, num, other_cond=''):
    sql_text = 'select max(tmp.resumeid) as cnum from (' \
               'select DISTINCT resumeid from resume_adjunct_beta ' \
               'where resumeid>{} {} order by resumeid limit {}) as tmp' \
               ''.format(resumeid, other_cond, num)
    results = sql_utils.query_by_sql(sql_text)
    maxid = results[0][0]
    if maxid is None:
        return None, None
    results = sql_utils.query_by_sql(
        'select t1.resumeid ,t1.colname,t1.labelid,t2.label,t1.word,t1.adjunct,t2.important '
        'from resume_adjunct_beta t1 inner join resume_label_store t2 '
        'where t1.labelid = t2.id and t1.resumeid>{} and t1.resumeid<={} {} '
        'group by resumeid,colname,labelid,word,adjunct '
        'order by resumeid,colname,labelid,word,adjunct'
        ''.format(resumeid, maxid, other_cond))
    if len(results) == 0:
        return None, None
    return results, results[-1][0]


def clear(taskid):
    sql_utils.clear_table_condition('resume_adjunct_beta', "taskid={}".format(taskid))


def close():
    sql_utils.close()

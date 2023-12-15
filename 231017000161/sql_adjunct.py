import sql_utils


def insert(data, taskid):
    col_names = 'resumeid,colname,labelid,sentence,word,adjunct,adjunctpos,times,unit,taskid'
    indata = []
    for i in range(len(data["strength"])):
        indata_dict = {}
        indata_dict['resumeid'] = data['resumeid']
        indata_dict['colname'] = data['colname']
        indata_dict['labelid'] = data['labelid']
        indata_dict['sentence'] = data['sentence']
        indata_dict['word'] = data['word']
        indata_dict['adjunct'] = data['strength'][i]
        indata_dict['adjunctpos'] = data['strengthpos'][i]
        indata_dict['times'] = data['times'][i]
        indata_dict['unit'] = data['unit'][i]
        indata_dict['taskid'] = taskid
        indata.append(indata_dict)
    sql_utils.insert_table_multi('resume_adjunct', indata, col_names)


def queryBatch(resumeid, num, col_names='resumeid,colname,labelid,sum(times) as reducenum,taskid',
               where_clause=' group by resumeid,colname,labelid order by resumeid,colname,labelid'):
    sql_text = 'select max(tmp.resumeid) as cnum from (select DISTINCT resumeid from resume_adjunct where resumeid>' + str(
        resumeid) + ' order by resumeid limit ' + str(
        num) + ') as tmp'
    results = sql_utils.query_by_sql(sql_text)
    maxid = results[0][0]
    if maxid is None:
        return None, None
    results = sql_utils.query_table('resume_adjunct', col_names,
                                    'resumeid>' + str(resumeid) + ' and resumeid<=' + str(maxid) + where_clause)
    if len(results) == 0:
        return None, None
    return results, results[-1][0]


def clear(taskid):
    sql_utils.clear_table_condition('resume_adjunct', "taskid={}".format(taskid))


def close():
    sql_utils.close()

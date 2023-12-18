import sql_utils


def insert(data):
    col_names = 'resumeid,scholarship,ability,reward,skill,experience,certify,taskid'
    sql_utils.insert_table_multi('resume_radar', data, col_names)


def clear(taskid):
    sql_utils.clear_table_condition('resume_radar', "taskid={}".format(taskid))


def close():
    sql_utils.close()

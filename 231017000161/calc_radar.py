import sql_adjunct_reduce
import sql_radar
import sql_utils

BATCH_NUMBER = 300

def calc_radar(taskid):
    sql_radar.clear(taskid)
    start_i = -1
    lastResumeid = ''
    i = 1
    while True:
        print('{} batch: startid={}'.format(i, start_i))
        i += 1
        (datas, start_i) = sql_adjunct_reduce.queryBatch(start_i, BATCH_NUMBER,
                                                         other_cond="and taskid={}".format(taskid))
        if datas is None:
            break

        saveData = []
        for data in datas:
            if str(data[0]) != lastResumeid:
                lastResumeid = str(data[0])
                saveData.append(
                    {"resumeid": lastResumeid, 'scholarship': 0, 'reward': 0,
                     'certify': 0, 'skill': 0, 'experience': 0, 'ability': 0,
                     "taskid": taskid})
            saveData[-1][fieldnameConvert(data[1])] = data[2]

        sql_radar.insert(saveData)

    uniform('scholarship', taskid)
    uniform('reward', taskid)
    uniform('certify', taskid)
    uniform('skill', taskid)
    uniform('experience', taskid)
    uniform('ability', taskid)


def uniform(colname, taskid):
    sql_utils.update_table_statement('resume_radar',
                                     ' {} = POWER({},4/5)'.format(colname, colname),
                                     ' taskid={} '.format(taskid))

    max = sql_utils.query_by_sql('select max({}) from resume_radar where taskid={}'
                                 ''.format(colname, taskid))
    maxscore = 30
    maxscore += max[0][0]
    sql_utils.update_table_statement('resume_radar',
                                     ' {} = ROUND(({}+{})*100/{},2)'
                                     ''.format(colname, colname, maxscore / 10.0, maxscore * 11.0 / 10),
                                     ' taskid={} '.format(taskid))


def fieldnameConvert(name):
    if name == '学术':
        return 'scholarship'
    elif name == '获奖':
        return 'reward'
    elif name == '认证':
        return 'certify'
    elif name == '才艺':
        return 'skill'
    elif name == '经历':
        return 'experience'
    elif name == '能力':
        return 'ability'


if __name__ == '__main__':
    calc_radar(-1)
    sql_radar.close()

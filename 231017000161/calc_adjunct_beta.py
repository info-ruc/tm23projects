import sql_adjunct
import sql_adjunct_beta

BATCH_NUMBER = 300


def calc_adjunct_beta(taskid):
    sql_adjunct_beta.clear(taskid)
    start_i = -1
    i = 1
    while True:
        print('{} batch: startid={}'.format(i, start_i))
        i += 1
        (datas, start_i) = sql_adjunct.queryBatch(start_i, BATCH_NUMBER,
                                                  'resumeid,colname,labelid,word,adjunct,sum(times) as times,taskid',
                                                  ' and taskid={} group by resumeid,colname,labelid,word,adjunct '
                                                  'order by resumeid,colname,labelid,word,adjunct'
                                                  ''.format(taskid))
        if datas is None:
            break
        sql_adjunct_beta.insert(datas)


if __name__ == '__main__':
    calc_adjunct_beta(-1)

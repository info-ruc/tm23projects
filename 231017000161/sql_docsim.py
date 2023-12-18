import sql_utils
from sql_labelStore import getAllLabelStore

origin_table_name = "resume_docsim_origin"
table_name = "resume_docsim"


def integration(taskid, dev=False):
    sql_utils.clear_table_condition(table_name, "taskid={}".format(taskid))
    rule = getAllLabelStore()
    if dev:
        for key in rule:
            print(key, rule[key])
    data_list = sql_utils.query_table(table=origin_table_name,
                                      col_names="resumeid,colname,similarity",
                                      condition="taskid={} order by resumeid".format(taskid))
    # sql查询出的结果经过resumeid排序，同一人的数据连续出现
    id_tmp = ""
    similarity = {}
    for data in data_list:
        # print(data)
        if data[0] != id_tmp:
            if similarity.__len__() > 0:
                data_line = {"resumeid": id_tmp, "similarity": similarity,
                             "taskid": taskid}
                # print(data_line)
                sql_utils.insert_table(table=table_name, data_line=data_line,
                                       col_names="resumeid,similarity,taskid", commit=False)
            id_tmp = data[0]
            similarity = {}
        sim_list = eval(data[2])
        for rule_line, sim in zip(rule[data[1]], sim_list):
            if rule_line[2] not in similarity:
                similarity[rule_line[2]] = sim
            else:
                similarity[rule_line[2]] += sim

    if similarity.__len__() > 0:
        data_line = {"resumeid": id_tmp, "similarity": similarity,
                     "taskid": taskid}
        # print(data_line)
        sql_utils.insert_table(table=table_name, data_line=data_line,
                               col_names="resumeid,similarity,taskid", commit=False)
    sql_utils.force_commit()


def integration_bak():
    rule = getAllLabelStore()
    # for key in rule:
    #     print(key, rule[key])
    data_list = sql_utils.query_table(table=table_name,
                                      col_names="resumeid,similarity_CA,similarity_CC,similarity_CO,"
                                                "similarity_COMPETENCY_ADV,similarity_OTHER_AWARD,"
                                                "similarity_POSITION,similarity_SCHOLARSHIP")
    col2id = {"CA": 1, "CC": 2, "CO": 3, "COMPETENCY_ADV": 4, "OTHER_AWARD": 5, "POSITION": 6, "SCHOLARSHIP": 7}
    for data in data_list:
        similarity = {}
        for key in rule:
            if data[col2id[key]] is None:
                for rule_line in rule[key]:
                    similarity[rule_line[2]] = 0
            else:
                # print(data[col2id[key]])
                sim_list = eval(data[col2id[key]])
                for rule_line, sim in zip(rule[key], sim_list):
                    similarity[rule_line[2]] = sim
        sql_utils.update_table(table=table_name, data_line={"similarity": similarity}, col_names="similarity",
                               condition='resumeid=' + str(data[0]), commit=False)
    sql_utils.force_commit()


def uniform():
    max = sql_utils.query_by_sql('select max(score) from resume_docsim')
    maxscore = 1
    if max[0][0] != 0:
        maxscore = max[0][0]
    sql_utils.update_table_statement('resume_docsim', ' score = ROUND(score*100/' + str(maxscore) + ',2)', ' 1=1 ')


def update_score(taskid):
    sql = 'update resume_docsim set score = (' \
          'select sum(scores) from resume_adjunct_reduce ' \
          'where resume_adjunct_reduce.taskid={} ' \
          'and resume_adjunct_reduce.taskid=resume_docsim.taskid ' \
          'and resume_adjunct_reduce.resumeid = resume_docsim.resumeid ' \
          'GROUP BY resumeid)'.format(taskid)
    sql_utils.run(sql)
    sql_utils.force_commit()
    print("++++++++++++++++++++++++++++++++")
    uniform()


def update_score_rank(taskid, data_list=None):
    # result: list (id,score)
    if data_list is None:
        data_list = sql_utils.query_table(table=table_name, col_names="id,score",
                                          condition="taskid={}".format(taskid))
    data_list = sorted(data_list, key=lambda x: (x[1], x[0]), reverse=True)
    # print_cross_platform(result[:10])
    pre_score = data_list[0][1] + 1
    num_pointer = -1
    num = 0
    for line in data_list:
        num += 1
        # print(num_pointer, pre_score)
        if pre_score != line[1]:
            num_pointer = num
            pre_score = line[1]
        data_line = {"scorerank": num_pointer}
        sql_utils.update_table(table_name, data_line, 'scorerank', 'id=' + str(line[0]), commit=False)
    sql_utils.force_commit()


def update_rank(taskid, data_list_before=None):
    # result: list (id,score)
    if data_list_before is None:
        data_list_before = sql_utils.query_table(table=table_name, col_names="id,similarity",
                                                 condition="taskid={}".format(taskid))
    data_list = [(data_line[0], eval(data_line[1])) for data_line in data_list_before]
    # print_cross_platform("data_list", data_list[:10], sep='\n')

    col_dict = {key: True for key in data_list[0][1]}
    ret = {data_line[0]: {} for data_line in data_list}
    for col in col_dict:
        data_col = [(data_line[0], data_line[1][col]) for data_line in data_list]
        data_col = sorted(data_col, key=lambda x: (x[1], x[0]), reverse=True)
        # print_cross_platform("data_col", data_col[:10], sep='\n')
        pre_score = data_col[0][1] + 1
        num_pointer = -1
        num = 0

        for data_line in data_col:
            num += 1
            # print(num_pointer, pre_score)
            if pre_score != data_line[1]:
                num_pointer = num
                pre_score = data_line[1]
            ret[data_line[0]][col] = num_pointer

    for id in ret:
        sql_utils.update_table(table_name, {'rank': ret[id]}, 'rank', 'id=' + str(id), commit=False)
    sql_utils.force_commit()


if __name__ == '__main__':
    # data_path = r"D:\Data\small" + "\\"
    # data_path = r"D:\Data" + "\\"
    data_path = r"./"

    # integration(-1, dev=True)
    # update_rank(-1)

    # update_score(-1)
    update_score_rank(-1)

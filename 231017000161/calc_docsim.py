import pickle

from gensim import corpora, similarities

from sql_labelStore import getAllLabelStore
from sql_source import getSourceById
from sql_utils import update_table, insert_table, force_commit, clear_table_condition
from utils import preprocessWithFlag


def get_rule(dev=False):
    rule = getAllLabelStore()
    if dev:
        for key in rule:
            print(key, rule[key])
    rule_doc = []
    rule_dict = {}
    for key in rule:
        # print(key, rule[key])
        rule_dict[key] = []
        for line in rule[key]:
            output, flag = preprocessWithFlag(line[3])
            rule_doc.append(output)
            rule_dict[key].append(output)
    return rule_dict, rule_doc


def get_dictionary(taskid, total, rule=None, rule_dict=None, batch_size=4096, start_id=0):
    if rule is None:
        return pickle.load(open('./process/all_{}.dic'.format(taskid), 'rb'))
    dictionary = corpora.Dictionary(rule)
    i = 1
    while True:
        print('func:get_dictionary {} batch: startid={}'.format(i, start_id))
        percent = int((i * 100.0 / total) * 22)
        if percent > 22:
            percent = 22
        i += 1
        source_dict, start_id = getSourceById(start_id, batch_size,
                                              other_cond='and taskid={}'.format(taskid))
        if source_dict is None:
            break
        doc = []
        for key in source_dict:
            for line in source_dict[key]:
                if (line[1] not in rule_dict):
                    continue
                # print(line)
                output, flag = preprocessWithFlag(line[2])
                doc.append(output)
                data_line = {"cut_content": ' '.join(output)}
                update_table('resume_source', data_line, "cut_content",
                             condition="resumeid={} and colname='{}' and taskid={}"
                                       "".format(line[0], line[1], taskid),
                             commit=False)
        force_commit()
        dictionary.add_documents(doc)
        print('----------------------------------\t+{}%\t{}%'.format(percent, percent))
    dictionary.filter_extremes(no_below=1, no_above=1, keep_n=None)
    dictionary.id2token = {dictionary.token2id[value]: value for value in dictionary.token2id}
    dictionary.save('./process/all_{}.dic'.format(taskid))
    return dictionary


def get_sim(rule_dict, dictionary, taskid, total, batch_size=4096, start_id=0):
    clear_table_condition('resume_docsim_origin', "taskid={}".format(taskid))
    rule_sim_model = {}
    for key in rule_dict:
        corpus = [dictionary.doc2bow(text) for text in rule_dict[key]]
        # print_cross_platform("corpus", corpus)
        rule_sim_model[key] = similarities.Similarity('-Similarity-index',
                                                      corpus, num_features=len(dictionary))
    # print(rule_sim_model)
    i = 1
    while True:
        print('func:get_sim {} batch: startid={}'.format(i, start_id))
        percent = int((i * 100.0 / total) * 21)
        if percent>21:
            percent=21
        i += 1
        source_dict, start_id = getSourceById(start_id, batch_size,
                                              col_names='resumeid,colname,cut_content',
                                              other_cond='and taskid={}'.format(taskid))
        if source_dict is None:
            break
        for key in source_dict:
            for line in source_dict[key]:
                if line[1] in rule_sim_model:
                    line[0] = str(line[0])
                    query_bow = dictionary.doc2bow(line[2].split(" "))
                    sim = list(rule_sim_model[line[1]][query_bow])
                    # data_line = {"similarity": sim}
                    # print(data_line)
                    # update_table('resume_source', data_line, "similarity",
                    #              condition="resumeid={} and colname='{}' and taskid={}"
                    #                        "".format(line[0], line[1], taskid),
                    #              commit=False)

                    data_line = {"resumeid": line[0], "colname": line[1],
                                 "similarity": sim, "taskid": taskid}
                    # print(data_line)
                    insert_table('resume_docsim_origin', data_line,
                                 "resumeid,colname,similarity,taskid", commit=False)
        force_commit()
        print('----------------------------------\t+{}%\t{}%'.format(percent, 22 + percent))


def calc_docsim(taskid, total):
    rule_dict, rule_doc = get_rule()
    #print(rule_doc)

    #dictionary = get_dictionary(taskid, total, batch_size=128, rule=rule_doc, rule_dict=rule_dict)  # 生成字典
    dictionary = get_dictionary(taskid, total)  # 生成字典
    # dictionary = get_dictionary(taskid)  # 读取字典
    print('----------------------------------\t+22%\t22%')

    get_sim(rule_dict, dictionary, taskid, total, batch_size=128)
    print('----------------------------------\t+21%\t43%')


if __name__ == '__main__':
    calc_docsim(taskid=-1)

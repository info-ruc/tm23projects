from sql_adjunct import insert, clear
from sql_labelStore import getAllLabelStore
from sql_source import getSourceAndSimById
from utils_ltp import sentence_splitter, dep_parser, pos_tag, segment

BATCH_NUMBER = 100

# 获取标签库
labelstore = getAllLabelStore()


def loadNumber():
    lines = open('./source/number.txt', 'r', encoding='utf-8').readlines()
    numberdict = {}
    for line in lines:
        word = line.split()
        numberdict[word[0].strip()] = word[1].strip()
    return numberdict


numberdict = loadNumber()


def loadVerbStop():
    lines = open('./source/verbstop.txt', 'r', encoding='utf-8').readlines()
    numberdict = set([])
    for line in lines:
        numberdict.add(line.strip())
    return numberdict


verbStop = loadVerbStop()


def calc_adjunct(taskid, total):
    clear(taskid)
    start_i = -1
    i = 1
    while True:
        print('func:calc_adjunct {} batch: startid={}'.format(i, start_i))
        percent = int((i * 100.0 / total) * 33)
        if percent > 33:
            percent = 33
        i += 1
        (sources, start_i, sim) = getSourceAndSimById(start_i, BATCH_NUMBER,
                                                      other_cond='and taskid={}'.format(taskid))
        # sources=[（resumeid,colname,norm_content）]
        if sources is None:
            break
        for (index, key) in enumerate(sources):
            if sim[index][0] != key:
                print('error:similarity not belong resumeid line')
            calc_adjunct_resume(sources[key], eval(sim[index][1]), taskid)
        print('----------------------------------\t+{}%\t{}%'.format(percent, 50 + percent))


def calc_adjunct_resume(source, simdict, taskid):
    # process each source line
    for line in source:
        sentences = sentence_splitter(line[2])
        for sentence in sentences:
            calc_adjunct_sentence(line[0], line[1], sentence, simdict, taskid)


def calc_adjunct_sentence(resumeid, colname, sentence, simdict, taskid):
    if colname not in labelstore:
        return
    label_list = labelstore[colname]
    for label in label_list:
        # only the flag need adj calc is 1 will enter the functoin
        if label[6] == 1 or (label[6] == 0 and simdict[label[2]] != 0):
            calc_adjunct_sentence_label(resumeid, colname, label, sentence, simdict, taskid)


def calc_adjunct_sentence_label(resumeid, colname, labelline, sentence, simdict, taskid):
    # if simdict[labelline[2]]==0:
    #     return
    label_words = labelline[3].split()
    for label_word in label_words:
        label_word = label_word.strip()
        if label_word == '':
            continue
        if sentence.find(label_word) < 0:
            continue
        calc_adjunct_sentence_word(resumeid, colname, labelline, label_word, sentence, simdict, taskid)


def calc_adjunct_sentence_word(resumeid, colname, labeline, label_word, sentence, simdict, taskid, debug_flag=False):
    if labeline != '':
        labelid = labeline[0]
    else:
        labelid = ''
    words = segment(sentence)
    postags = pos_tag(words)
    arcs = dep_parser(words)
    if debug_flag:
        print('分词结果：', ' '.join(str((value, index)) for (index, value) in enumerate(words)))
        print('词性标注结果：', ' '.join(str((index, value)) for (index, value) in enumerate(postags)))
        print('依存句法分析结果：',
              "\t".join("%d-%d:%s" % (index, arc.head - 1, arc.relation) for (index, arc) in enumerate(arcs)))

    target_pos = [index for (index, value) in enumerate(words) if value == label_word]
    # print('target:',' '.join("%d" % (pos+1) for pos in target_pos))

    relation_group = []
    relation_group_extend = []
    COO_set = []
    for i in range(len(words)):
        relation_group.append(set([i]))
        relation_group_extend.append(set([i]))
        COO_set.append(set([]))

    for (index, arc) in enumerate(arcs):
        if arc.relation == 'ATT' or arc.relation == 'VOB' or arc.relation == 'ADV' or arc.relation == 'CMP':
            # 合并具有XX关系的词语成为一个聚簇
            # set group base
            bak_index_relation = relation_group[index].copy()
            bak_head_relation = relation_group[arc.head - 1].copy()
            mix = relation_group[index] | relation_group[arc.head - 1]
            for pos in bak_index_relation:
                relation_group[pos] = mix
            for pos in bak_head_relation:
                relation_group[pos] = mix
            # set group extend: including coo relation
            bak_index_relation_extend = relation_group_extend[index].copy()
            bak_head_relation_extend = relation_group_extend[arc.head - 1].copy()
            mix = relation_group_extend[index] | relation_group_extend[arc.head - 1]
            for pos in bak_index_relation_extend:
                relation_group_extend[pos] = mix
            for pos in bak_head_relation_extend:
                relation_group_extend[pos] = mix
        if arc.relation == 'COO':
            if len(COO_set[index]) == 0 and len(COO_set[arc.head - 1]) == 0:
                bak_index_relation_extend = relation_group_extend[index].copy()
                bak_head_relation_extend = relation_group_extend[arc.head - 1].copy()
                mix = relation_group_extend[index] | relation_group_extend[arc.head - 1]
                for pos in bak_index_relation_extend:
                    relation_group_extend[pos] = mix
                    COO_set[pos].update([index, arc.head - 1])
                for pos in bak_head_relation_extend:
                    relation_group_extend[pos] = mix
                    COO_set[pos].update([index, arc.head - 1])
            else:
                if index in COO_set[index] or (arc.head - 1) in COO_set[arc.head - 1]:
                    bak_index_relation_extend = relation_group_extend[index].copy()
                    bak_head_relation_extend = relation_group_extend[arc.head - 1].copy()
                    mix = relation_group_extend[index] | relation_group_extend[arc.head - 1]
                    for pos in bak_index_relation_extend:
                        relation_group_extend[pos] = mix
                        COO_set[pos].update([index, arc.head - 1])
                    for pos in bak_head_relation_extend:
                        relation_group_extend[pos] = mix
                        COO_set[pos].update([index, arc.head - 1])

    strength = []
    strength_pos = []
    times = []
    unit = []
    for (idx, pos) in enumerate(target_pos):
        strength.append('')
        times.append('')
        unit.append('')
        strength_pos.append(1000)
        group_at_pos = list(relation_group[pos])
        # print('group:',group_at_pos)
        v_flag = 0
        for index in sorted(group_at_pos):
            if index == pos:
                continue
            if index in COO_set[index]:
                continue

            # 处理词语修饰成分开头，“三等” 奖学金
            if index != strength_pos[idx] + 1 or v_flag == 1:
                # 1.most possible: b as the adjunct  省级奖学金
                if postags[index] == 'b' and abs(pos - index) < abs(pos - strength_pos[idx]):
                    strength[idx] = words[index]
                    strength_pos[idx] = index
                # 2. possible: a as the adjunct  熟练使用c
                if postags[index] == 'a' and abs(pos - index) < abs(pos - strength_pos[idx]):
                    strength[idx] = words[index]
                    strength_pos[idx] = index
                # 3. possible:  n as the adjunct  国家奖学金
                if postags[index].startswith('n') and not postags[index].startswith('nt') and abs(pos - index) < abs(
                                pos - strength_pos[idx]):
                    strength[idx] = words[index]
                    strength_pos[idx] = index
                # 4. possible: v as the adjunct : 精通c++
                if postags[index] == 'v' and words[index] not in verbStop:
                    if strength[idx] == '':
                        v_flag = 1
                        strength[idx] = words[index]
                        strength_pos[idx] = index
                    else:
                        if postags[strength_pos[idx]] == 'v' and abs(pos - index) < abs(pos - strength_pos[idx]):
                            v_flag = 1
                            strength[idx] = words[index]
                            strength_pos[idx] = index
            # 处理词语修饰成分后续拼接，“省级 三等” 奖学金
            else:
                # 1.most possible: b as the adjunct  省级奖学金
                if postags[index] == 'b' and abs(pos - index) < abs(pos - strength_pos[idx]):
                    strength[idx] = strength[idx] + words[index]
                    strength_pos[idx] = index
                # 2. possible: a as the adjunct  熟练使用c
                if postags[index] == 'a' and abs(pos - index) < abs(pos - strength_pos[idx]):
                    strength[idx] = strength[idx] + words[index]
                    strength_pos[idx] = index
                # 3. possible:  n as the adjunct  国家奖学金
                if postags[index].startswith('n') and not postags[index].startswith('nt') and abs(pos - index) < abs(
                                pos - strength_pos[idx]):
                    strength[idx] = strength[idx] + words[index]
                    strength_pos[idx] = index
                # 4. possible: v as the adjunct : 精通c++
                if postags[index] == 'v' and words[index] not in verbStop:
                    if strength[idx] == '':
                        v_flag = 1
                        strength[idx] = words[index]
                        strength_pos[idx] = index
                    else:
                        if postags[strength_pos[idx]] == 'v' and abs(pos - index) < abs(pos - strength_pos[idx]):
                            v_flag = 1
                            strength[idx] = words[index]
                            strength_pos[idx] = index

            # index是数词，index+1是量词，则记录。“三次” 奖学金
            if postags[index] == 'm' and index < len(words) - 1 and postags[index + 1] == 'q':
                times[idx] = words[index]
                unit[idx] = words[index + 1]

        # 若未查询到修饰成分，尝试使用并列关系ＣＯＯ的修饰成分（这时多词语修饰极少见，无需拼接）。
        if strength[idx] == '':
            group_at_pos_extend = list(relation_group_extend[pos])
            for index in sorted(group_at_pos_extend):
                if index == pos:
                    continue
                if index in COO_set[index]:
                    continue
                if pos < index:
                    continue
                # 1.most possible: b as the adjunct  省级奖学金
                if postags[index] == 'b' and abs(pos - index) < abs(pos - strength_pos[idx]):
                    strength[idx] = words[index]
                    strength_pos[idx] = index
                # 2.possible: a as the adjunct  熟练使用c
                if postags[index] == 'a' and abs(pos - index) < abs(pos - strength_pos[idx]):
                    strength[idx] = words[index]
                    strength_pos[idx] = index
                # 3.possible:  n as the adjunct  国家奖学金
                if postags[index].startswith('n') and not postags[index].startswith('nt') and abs(pos - index) < abs(
                                pos - strength_pos[idx]):
                    strength[idx] = words[index]
                    strength_pos[idx] = index
                # 4.possible: v as the adjunct : 精通c++
                if postags[index] == 'v' and words[index] not in verbStop:
                    if strength[idx] == '':
                        v_flag = 1
                        strength[idx] = words[index]
                        strength_pos[idx] = index
                    else:
                        if postags[strength_pos[idx]] == 'v' and abs(pos - index) < abs(pos - strength_pos[idx]):
                            v_flag = 1
                            strength[idx] = words[index]
                            strength_pos[idx] = index

    save_data = {"resumeid": resumeid,
                 "colname": colname,
                 "labelid": labelid,
                 "sentence": sentence,
                 "word": label_word,
                 "strength": strength,
                 "strengthpos": strength_pos,
                 "times": translateNum(times),
                 "unit": unit}
    if labeline != '' and labeline[6] == 0:
        save_data["strength"] = ['']
        save_data["strengthpos"] = ['0']
        save_data["times"] = [1]
        save_data["unit"] = ['']

    if debug_flag:
        print(label_word, sentence, '----------------------------')
        print('strength:', strength, strength_pos)
        print('times:', translateNum(times))
        print('unit:', unit)
    else:
        insert(save_data, taskid)


def findLeastNegative(pos, datas):
    proc = []
    for data in datas:
        if data > pos:
            proc.append(data)
    proc = sorted(proc)
    return proc[0]


def translateNum(datas):
    retdata = []
    for data in datas:
        if data in numberdict:
            retdata.append(int(numberdict[data]))
        else:
            retdata.append(1)
    return retdata


if __name__ == '__main__':

    flag = False

    if not flag:
        calc_adjunct(-1)
    # print('--------------------------------------------------')

    if flag:
        # calc_adjunct_sentence_word('','','','三好学生','本人2017年荣获校级三好学生一次、两次省级三好学生。',True)
        # # print('--------------------------------------------------')
        # calc_adjunct_sentence_word('','','','奖学金','2009年国家奖学金，两次2019年省级奖学金,2023年周大福奖学金,2009年级奖学金两层',True)
        # # print('--------------------------------------------------')
        # calc_adjunct_sentence_word('','','','奖学金', '2006 - 2007学年获校一等奖学金、“ 三好学生”、国家“励志”奖学金,2007 - 2008学年获校一等奖学金、校“尤洛卡”奖学金、“优秀学生标兵”',True)
        # # print('--------------------------------------------------')
        # #calc_adjunct_sentence_word('EXCEL','本人计算机能力强、熟练使用MSOFFICE、精通WD及EXCEL、毕业设计使用SPSS专业软件进行金融分析')12月
        # calc_adjunct_sentence_word('','','','奖学金', '获得奖学金2次',True)
        # calc_adjunct_sentence_word('', '', '', 'dreamweaver', '擅长新闻专题策划、采访及稿件写作、并熟悉媒介所需软件的操作应用、如方正飞腾、photoshop、dreamweaver、绘声绘影、快马飞编等',True)
        calc_adjunct_sentence_word('', '', '', 'Hadoop', '掌握 Hadoop 生态圈主流的大数据相关组件技术及工作原理', '', True)
        # calc_adjunct_sentence_word('', '', '', 'java', '精通C++、java。','',True)
        # calc_adjunct_sentence_word('', '', '', '奖学金', '2007学年获国家奖学金,国家励志奖学金', True)

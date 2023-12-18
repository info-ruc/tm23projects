'''
Best practice 01

Put all constants in one file, and protect them from changing value.

'''
# File name: constants.py


class Const(object):
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:  # 判断是否已经被赋值，如果是则报错
            raise self.ConstError("Can't change const.%s" % name)
        if not name.isupper():  # 判断所赋值是否是全部大写，用来做第一次赋值的格式判断，也可以根据需要改成其他判断条件
            raise self.ConstCaseError('const name "%s" is not all supercase' % name)

        self.__dict__[name] = value


const = Const()
# files
const.PROJECT_DIR = "D:\\xueqing\\workplace\\lstm\\Time-Series-Prediction-with-LSTM"
const.EXPERIMENTS_DIR = const.PROJECT_DIR + "/data/experiments"
const.DATA_DIR = const.PROJECT_DIR + "/data"
# neural networks flag
const.FLAG_NN_BP = 0
const.FLAG_NN_RNN = 1
const.FLAG_NN_LSTM = 2
const.FLAG_NN_STRING = ['BP', 'RNN', 'LSTM']
# train & validate & test
const.TRAIN_SCALE = 0.80
const.VALIDATION_SCALE = 0.25
const.LOOK_BACK = 30
const.OUTPUT = 1


# TEST
# File name: test.py
# from constants import const
# print(const.MY_CONSTANT)
# const.MY_CONSTANT = 2  # 此处尝试再赋值会触发ConstError


# --------------------- 
# 作者：python012 
# 来源：CSDN 
# 原文：https://blog.csdn.net/python012/article/details/80010490 
# 版权声明：本文为博主原创文章，转载请附上博文链接！
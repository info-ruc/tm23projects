def __init__(self,stop_word_file=None,features_vocabulary=None):
        self.features_vocabulary = features_vocabulary
        self.stop_vocab_dict = {}  
        if stop_word_file is not None:
            self.stop_vocab_dict = self._get_stop_words(stop_word_file)

def text_to_feature_matrix(self,words,vocabulary=None,threshold =10):
        cv = CountVectorizer()
        if vocabulary is None:
            cv.fit(words)
        else:
            cv.fit(vocabulary)
        words_to_vect = cv.transform(words)
        words_to_matrix = pd.DataFrame(words_to_vect.toarray())  
        print(words_to_matrix.shape)
        selected_features = []
        selected_features_index = []
        for key,value in cv.vocabulary_.items():
            if words_to_matrix[value].sum() >= threshold:  
                selected_features.append(key)
                selected_features_index.append(value)

def get_email_words(self,email_path, max_email = 600):
        self.emails = email_path
        if os.path.isdir(self.emails):
            emails = os.listdir(self.emails)
            is_dir = True
        else:
            emails = [self.emails,]
            is_dir = False
        count = 0
        all_email_words = []
        for email in emails:
            if count >= max_email:  
                break
            if is_dir:
                email_path = os.path.join(self.emails,email)
            email_words = self._email_to_words(email_path)
            all_email_words.append(' '.join(email_words))
            count += 1
        return all_email_words

def _email_to_words(self, email):
        retrun:words_list  
        email_words = []
        with open(email, 'rb') as pf:
            for line in pf.readlines():
                line = line.strip().decode('gbk','ignore')
                if not self._check_contain_chinese(line):  
                    continue
                word_list = jieba.cut(line, cut_all=False)  
                for word in word_list:
                    if word in self.stop_vocab_dict or not self._check_contain_chinese(word):
                        continue  
                    email_words.append(word)
            return email_words

def _get_stop_words(self,file):
        stop_vocab_dict = {}
        with open(file,'rb') as pf:
            for line in pf.readlines():
                line = line.decode('utf-8','ignore').strip()
                stop_vocab_dict[line] = 1
        return stop_vocab_dict
def _check_contain_chinese(self,check_str):
        for ch in check_str:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False


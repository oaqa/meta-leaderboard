import gzip
import ast
import os

import logging
from logging import config
import json
import numpy as np
from nltk.translate import bleu
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
#from PyRouge.pyrouge import Rouge
#from pyrouge import Rouge155
from rouge import Rouge 


logging.config.fileConfig('logging.ini')
logger = logging.getLogger('qaLogger')


class Formatter(object):
    def __init__(self, filePath, dataset_type):
        self.filePath = filePath
        self.dataset_type = dataset_type
        if dataset_type == 'msmarco':
            self.reformatMsmarco()
        elif dataset_type == 'searchqa':
            self.reformatSearchqa()
        elif dataset_type == 'bioasq':
            self.reformatBioasq()
        elif dataset_type == 'quasar-s':
            self.reformatQuasar_s()
        elif dataset_type == "quasar-t":
            self.reformatQuasar_t()

    def reformatMsmarco(self):
        formattedData = {}
        formattedData['origin'] = self.dataset_type
        allQsns = []
        with gzip.open(self.filePath) as f:
            for line in f:
                cur_qsn = {}
                d = ast.literal_eval(line)
                #print d.keys() # ['query_type', 'passages', 'query_id', 'answers', 'query']
                cur_qsn['query'] = d['query']
                cur_qsn['contexts'] = d['passages']
                cur_qsn['answer_type'] = d['query_type']
                cur_qsn['answers'] = d['answers']
                cur_qsn['id'] = d['query_id']
                cur_qsn['meta_data'] = {}
                allQsns.append(cur_qsn)
        formattedData['questions'] = allQsns
        return formattedData

    def reformatSearchqa(self):
        formattedData = {}
        formattedData['origin'] = self.dataset_type
        allQsns = []
        folderPath = self.filePath
        allFiles = [f for f in os.listdir(folderPath)]
        for f in allFiles:
            cur_qsn = {}
            data = json.load(open(os.path.join(folderPath, f), 'r'))
            #print data.keys() # [u'category', u'search_results', u'air_date', u'question', u'value', u'round', u'answer', u'id', u'show_number']
            cur_qsn['query'] = data['question']
            cur_qsn['contexts'] = data['search_results']
            cur_qsn['answer_type'] = data['category']
            cur_qsn['answers'] = data['answer']
            cur_qsn['id'] = data['id']
            cur_qsn['meta_data'] = {}
            cur_qsn['meta_data']['air_date'] = data['air_date']
            cur_qsn['meta_data']['value'] = data['value']
            cur_qsn['meta_data']['round'] = data['round']
            cur_qsn['meta_data']['show_number'] = data['show_number']
            allQsns.append(cur_qsn)
        formattedData['questions'] = allQsns
        return formattedData


    def reformatBioasq(self):
        formattedData = {}
        formattedData['origin'] = self.dataset_type
        allQsns = []
        infile = open(self.filePath, 'r')
        data = json.load(infile)
        for (i, question) in enumerate(data['questions']):
            cur_qsn = {}
            #print question.keys() #[u'body', u'documents', u'ideal_answer', u'type', u'id', u'snippets']
            cur_qsn['query'] = question['body']
            cur_qsn['contexts'] = {}
            cur_qsn['contexts']['long_snippets'] = question['snippets']
            cur_qsn['contexts']['urls'] = question['documents']
            cur_qsn['answer_type'] = question['type']
            cur_qsn['exact_answer'] = question['exact_answer']
            cur_qsn['ideal_answer'] = question['ideal_answer']
            cur_qsn['id'] = question['id']
            cur_qsn['meta_data'] = {}
            allQsns.append(cur_qsn)
        formattedData['questions'] = allQsns
        return formattedData

    def reformatQuasar_s(self):
        formattedData = {}
        formattedData['origin'] = self.dataset_type
        allQsns = []

        #loading context files
        longFile = "../curtis.ml.cmu.edu/datasets/quasar/quasar-s/contexts/long/dev_contexts.json.gz"
        with gzip.open(longFile) as lf:
            longClines = lf.readlines()
        lf.close()

        shortFile = "../curtis.ml.cmu.edu/datasets/quasar/quasar-s/contexts/short/dev_contexts.json.gz"
        with gzip.open(shortFile) as sf:
            shortClines = sf.readlines()
        lf.close()

        with gzip.open(self.filePath) as f:
            i = -1
            for line in f:
                i += 1
                cur_qsn = {}
                d = ast.literal_eval(line)
                #print d.keys() #['answer', 'question', 'uid', 'tags']
                cur_qsn['query'] = d['question']
                cur_qsn['id'] = d['uid']
                cur_qsn['exact_answer'] = d['answer']
                cur_qsn['ideal_answer'] = []
                tags = d['tags']
                print cur_qsn['id']
                #print tags #['yes-answer-long', 'yes-answer-short']
                for tag in tags:
                    if tag == "yes-answer-long":
                        d = ast.literal_eval( longClines[i].strip() )
                        #print d.keys() #['contexts', 'uid']
                        cur_contexts_long, cur_contexts_scores_long = self.parseContext(d)
                    elif tag == "yes-answer-short":
                        d = ast.literal_eval( shortClines[i].strip() )
                        cur_contexts_short, cur_contexts_scores_short = self.parseContext(d)
                cur_qsn['contexts'] = {}
                cur_qsn['contexts']['long_snippets'] = cur_contexts_long
                cur_qsn['contexts']['long_scores'] = cur_contexts_scores_long
                cur_qsn['contexts']['short_snippets'] = cur_contexts_short
                cur_qsn['contexts']['short_scores'] = cur_contexts_scores_short
                allQsns.append(cur_qsn)
            formattedData['questions'] = allQsns
            return formattedData
                        
                

                        #cur_qsn['contexts']
                        
                raw_input()

    def parseContext(self, d):
        cur_contexts = []
        cur_contexts_scores = []
        for el in d['contexts']:
            cur_contexts.append(el[1])
            cur_contexts_scores.append(el[0])
        return cur_contexts, cur_contexts_scores


       

if __name__ == '__main__':
    #scenario = "train/dev/test"
    msmarcoFilePath = ["/Users/khyathi/Projects/QA_datasets/msmarco/dev_v1.1.json.gz", 'msmarco']
    searchqaFilePath = ["/Users/khyathi/Projects/QA_datasets/SearchQA/qacrawler/json_files/val", 'searchqa']
    bioasqFilePath = ["/Users/khyathi/temp2/BioasqArchitecture/submission.json", 'bioasq']
    quasar_sFilePath = ["/Users/khyathi/Projects/QA_datasets/curtis.ml.cmu.edu/datasets/quasar/quasar-s/questions/dev_questions.json.gz", 'quasar-s']
    quasar_tFilePath = ["/Users/khyathi/Projects/QA_datasets/curtis.ml.cmu.edu/datasets/quasar/quasar-t/questions/dev_questions.json.gz", 'quasar-t']
    #formatterInstance = Formatter(msmarcoFilePath[0], msmarcoFilePath[1])
    #formatterInstance = Formatter(searchqaFilePath[0], searchqaFilePath[1])
    #formatterInstance = Formatter(bioasqFilePath[0], bioasqFilePath[1])
    formatterInstance = Formatter(quasar_sFilePath[0], quasar_sFilePath[1])
    #formatterInstance = Formatter(quasar_tFilePath[0], quasar_tFilePath[1])




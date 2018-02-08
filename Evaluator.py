import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import logging
from logging import config
import json
import numpy as np
import string
from nltk import bleu
#from nltk.translate import bleu
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
#from PyRouge.pyrouge import Rouge
#from pyrouge import Rouge155
from rouge import Rouge 


#logging.config.fileConfig('logging.ini')
#logger = logging.getLogger('qaLogger')


class Evaluator(object):
    def __init__(self):
        self.gold_msmarco_dev = "~/Projects/QA_datasets/common_pipeline/msmarco_dev_formatted.json"
        self.gold_searchqa_dev = "/Users/khyathi/Projects/QA_datasets/common_pipeline/searchqa_dev_formatted.json"
        self.gold_bioasq_train = "/Users/khyathi/Projects/QA_datasets/common_pipeline/bioasq_train_formatted.json"
        self.gold_quasars_dev = "~/Projects/QA_datasets/common_pipeline/quasar-s_dev_formatted.json"
        

    def loadFiles(self):
        self.systemData = json.load(open(self.systemFilePath, 'r'))
        self.cur_metaData = self.systemData['meta_data']
        self.systemQuestions = self.systemData['questions']
        self.division = self.cur_metaData['division']
        self.cur_origin = self.systemData['origin']
        if self.cur_origin == 'msmarco':
            self.goldFilePath = self.gold_msmarco_dev
        elif self.cur_origin == 'searchqa':
            self.goldFilePath = self.gold_searchqa_dev
        elif self.cur_origin == 'bioasq':
            self.goldFilePath = self.gold_bioasq_train
        elif self.cur_origin == 'quasar-s':
            self.goldFilePath = self.gold_quasars_dev
        self.goldData = json.load(open(self.goldFilePath, 'r'))
        self.goldQuestions = self.goldData['questions']
        

    def performEvaluation(self, systemFilePath):
        self.systemFilePath = systemFilePath
        self.loadFiles()
        measuresList = ['rouge', 'bleu', 'precision', 'recall', 'f_measure', 'f1_match', 'accuracy']
        username = self.getUser()
        origin = self.cur_origin
        scoreDict = { 'bleu' : 'NA', 'rouge' : 'NA' , 'precision' : 'NA', 'recall' : 'NA', 'f_measure' : 'NA', \
                    'f1_match' : 'NA', 'accuracy' : 'NA', 'username': username , 'origin': origin}
        if origin == 'msmarco':
            #scoreDict['bleu'] = self.getBleu()
            scoreDict['rouge'] = self.getRouge()
            scoreDict['f1_match'] = self.getf1_match()
            scoreDict['accuracy'] = self.exact_match()
        elif origin == 'searchqa':
            scoreDict['accuracy'] = self.getAccuracy()
            scoreDict['f1_match'] = self.getf1_match()
        elif origin == 'bioasq':
            #scoreDict['bleu'] = self.getBleu()
            scoreDict['rouge'] = self.getRouge()
            scoreDict['f1_match'] = self.getf1_match()
            scoreDict['accuracy'] = self.getAccuracy()
        elif origin == 'quasar-s':
            scoreDict['accuracy'] =self.getAccuracy()
        elif origin == 'quasar-t':
            x=1
        f = open("scores.txt", "a")
        scoreString = username + "\t" + origin + "\t"
        for el in measuresList:
            scoreString += str(scoreDict[el]) + "\t"
        f.write(scoreString+"\n")
        f.close()
        '''index = 0
        for line in contents:
            w = line.strip()[1]
            if w <= score:
                break
            index += 1
        f.close()
        '''

        #contents.insert(index, str(filename)+" "+ str(score)+"\n")
        #f = open("scores.txt", "w")
        return scoreDict

    def getUser(self):
        username = self.cur_metaData['username']
        return username

    def getBleu(self):
        '''
        >>> from nltk import bleu
        >>> ref = 'let it go'.split()
        >>> hyp = 'let go it'.split()
        >>> bleu([ref], hyp)
        '''
        BSCORES = []
        for i in range(len(self.systemQuestions)):
            reference = self.goldQuestions[i]["ideal_answers"][0].split() #[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
            candidate = self.systemQuestions[i]["ideal_answers"][0].split()
            score = bleu([reference], candidate )
            BSCORES.append(score)
        bleu = np.mean(BSCORES)
        return bleu

    def getRouge(self):
        '''
        hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you saw on cnn student news"
        reference = "this page includes the show transcript use the transcript to help students with reading comprehension and vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests students ' knowledge of even ts in the news"
        rouge = Rouge()
        scores = rouge.get_scores(reference, hypothesis)
        '''
        RSCORES = []
        rouge = Rouge()
        for i in range(len(self.systemQuestions)):
            reference = str(self.goldQuestions[i]["ideal_answers"][0].encode('utf-8'))
            candidate = str(self.systemQuestions[i]["ideal_answers"][0].encode('utf-8'))
            scores = rouge.get_scores(candidate, reference)
            RSCORES.append(scores[0]['rouge-2']['f'])
        r2 = np.mean(RSCORES)
        return r2

    def getMRR(self): #should change this
        for i in range(len(self.systemQuestions)):
            rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
            rs = (np.asarray(r).nonzero()[0] for r in rs)
            mrr = np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])
        print mrr

    def getPRF(self):
        PRECISION = []
        RECALL = []
        F_MEASURE = []
        for i in range(len(self.systemQuestions)):
            y_pred = self.systemQuestions[i]['answers']
            y_true = self.goldQuestions[i]['answers']
            prf = precision_recall_fscore_support(y_true, y_pred, average='micro')
            PRECISION.append( prf[0] )
            RECALL.append( prf[1] )
            F_MEASURE.append( prf[2] )
        precision = np.mean(PRECISION)
        recall = np.mean(RECALL)
        f_measure = np.mean(F_MEASURE)
        return precision, recall, f_measure

    def clean_string(self, x):
        # lowercase, remove articles, remove punctuation, 
        # and return as a single string without whitespace
        pct = string.punctuation
        articles = ['a','an','the']
        toks = filter(lambda t:t not in articles, x.lower().split())
        return ''.join([''.join(filter(lambda c:c not in pct, list(t))) for t in toks])

    def exact_match(goldAnswer, predAnswer):
        xc1 = _clean_string(goldAnswer)
        xc2 = _clean_string(predAnswer)
        return xc1==xc2

    def getf1_match(self):
        F1_MATCH = []
        for i in range(len(self.systemQuestions)):
            goldAnswer = self.goldQuestions[i]['ideal_answers'][0]
            predAnswer = self.systemQuestions[i]['ideal_answers'][0]
            tok1 = set(map(lambda t:self.clean_string(t), goldAnswer.split()))
            tok2 = set(map(lambda t:self.clean_string(t), predAnswer.split()))
            l1 = len(tok1)
            l2 = len(tok2)
            ovr = len(tok1.intersection(tok2))
            if ovr==0: F1_MATCH.append(0)
            prec = float(ovr)/l2
            rec = float(ovr)/l1
            F1_MATCH.append( 2*prec*rec/(prec+rec) )
        f1_match = np.mean(F1_MATCH)
        return f1_match

    def getAccuracy(self):
        ACCURACY = []
        for i in range(len(self.systemQuestions)):
            y_pred = self.systemQuestions[i]['answers']
            y_true = self.goldQuestions[i]['answers']
        accuracy = accuracy_score(y_true, y_pred)
        print accuracy


if __name__ == '__main__':
    goldFilePath = "/Users/khyathi/temp2/BioasqArchitecture/submission.json"
    systemFilePath = "/Users/khyathi/temp2/BioasqArchitecture/submission.json"
    evaluatorInstance = Evaluator(goldFilePath, systemFilePath, 'bleu')

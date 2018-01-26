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


class Evaluator(object):
    def __init__(self, goldFilePath, systemFilePath, metric):
        self.goldFilePath = goldFilePath
        self.systemFilePath = systemFilePath
        self.metric = metric
        self.loadFiles()
        #self.getPRF()
        #self.getMRR()
        self.getRouge()

    def loadFiles(self):
        goldData = json.load(open(self.goldFilePath, 'r'))
        systemData = json.load(open(self.systemFilePath, 'r'))
        self.goldQuestions = goldData['questions']
        self.systemQuestions = systemData['questions']

    def getBleu(self):
        print "computing the bleu score"
        logger.info('computing bleu score')
        for i in range(len(self.systemQuestions)):
            reference = self.goldQuestions[i]["ideal_answer"].split() #[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
            candidate = self.systemQuestions[i]["ideal_answer"].split()
            score = bleu([reference], candidate )
            print score

    def getRouge(self):
        rouge = Rouge()
        for i in range(len(self.systemQuestions)):
            reference = str(self.goldQuestions[i]["ideal_answer"])
            candidate = str(self.systemQuestions[i]["ideal_answer"])
            scores = rouge.get_scores(candidate, reference)
            print scores

    def getMRR(self):
        print "computing mean reciprocal rank"
        ranking = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        rs = (np.asarray(r).nonzero()[0] for r in rs)
        mrr = np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])
        print mrr

    def getPRF(self):
        print "computing precison recall and f-measure"
        y_true = np.array(['ans1', 'ans1', 'ans3'])
        y_pred = np.array(['ans2', 'ans1', 'ans2'])
        prf = precision_recall_fscore_support(y_true, y_pred, average='micro')
        print prf

    def _clean_string(x):
        # lowercase, remove articles, remove punctuation, 
        # and return as a single string without whitespace
        toks = filter(lambda t:t not in articles, x.lower().split())
        return ''.join([''.join(filter(lambda c:c not in pct, list(t))) for t in toks])

    def exact_match(goldAnswer, predAnswer):
        xc1 = _clean_string(goldAnswer)
        xc2 = _clean_string(predAnswer)
        return xc1==xc2

    def f1_match(goldAnswer, predAnswer):
        tok1 = set(map(lambda t:_clean_string(t), goldAnswer.split()))
        tok2 = set(map(lambda t:_clean_string(t), predAnswer.split()))
        l1 = len(tok1)
        l2 = len(tok2)
        ovr = len(tok1.intersection(tok2))
        if ovr==0: return 0.
        prec = float(ovr)/l2
        rec = float(ovr)/l1
        return 2*prec*rec/(prec+rec)

    def accuracy(goldList, predList):
        goldList = [0, 2, 1, 3]
        predList = [0, 1, 2, 3]
        accuracy = accuracy_score(goldList, predList)
        print accuracy

       

if __name__ == '__main__':
    goldFilePath = "/Users/khyathi/temp2/BioasqArchitecture/submission.json"
    systemFilePath = "/Users/khyathi/temp2/BioasqArchitecture/submission.json"
    evaluatorInstance = Evaluator(goldFilePath, systemFilePath, 'bleu')

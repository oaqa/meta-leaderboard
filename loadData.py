class loadData(object):
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
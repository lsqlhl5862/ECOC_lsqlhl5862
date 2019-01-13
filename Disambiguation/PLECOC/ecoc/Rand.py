import numpy as np
from ecoc.BasePLECOC import BasePLECOC
from sklearn.svm import libsvm
from svmutil import *
from GetComplexity import *


class RandPLECOC(BasePLECOC):

    def __init__(self, estimator=libsvm, max_iter=2000, **params):
        BasePLECOC.__init__(self, estimator, **params)
        self.max_iter = max_iter
        self.num_class = None
        self.codingLength = None
        self.min_num_tr = None
        self.coding_matrix = None
        self.models = None
        self.performance_matrix = None
        self.params = params

    def create_coding_matrix(self, tr_data, tr_labels):
        num_tr = tr_data.shape[0]
        self.num_class = tr_labels.shape[0]
        self.codingLength = int(np.ceil(10 * np.log2(self.num_class)))
        self.min_num_tr = int(np.ceil(0.1 * num_tr))

        coding_matrix = None
        counter = 0
        tr_pos_idx = []
        tr_neg_idx = []

        # test code start
        # csv_path = 'csv/matrix_dump.csv'
        # if os.path.exists(csv_path):
        #     os.remove(csv_path)
        # test_code_matrix = pd.read_csv(csv_path, header=-1).values
        # for i in range(test_code_matrix.shape[1]):
        # test code end
        for i in range(self.max_iter):
            tmpcode = np.int8(np.random.rand(self.num_class) > 0.5)
            # tmpcode = test_code_matrix[:, i]
            tmp_pos_idx = []
            tmp_neg_idx = []
            for j in range(num_tr):
                if np.all((tr_labels[:, j] & tmpcode) == tr_labels[:, j]):
                    tmp_pos_idx.append(j)
                else:
                    if np.all((tr_labels[:, j] & np.int8(np.logical_not(tmpcode))) == tr_labels[:, j]):
                        tmp_neg_idx.append(j)
            num_pos = len(tmp_pos_idx)
            num_neg = len(tmp_neg_idx)

            if (num_pos+num_neg >= self.min_num_tr) and (num_pos >= 5) and (num_neg >= 5):
                counter = counter + 1
                tr_pos_idx.append(tmp_pos_idx)
                tr_neg_idx.append(tmp_neg_idx)
                coding_matrix = tmpcode if coding_matrix is None else np.vstack((coding_matrix, tmpcode))

            if counter == self.codingLength:
                break

        if counter != self.codingLength:
            raise ValueError('The required codeword length %s not satisfied', str(self.codingLength))
            self.codingLength = counter
            if counter == 0:
                raise ValueError('Empty coding matrix')
        # dump_matrix = pd.DataFrame(coding_matrix.T)
        # dump_matrix.to_csv(csv_path, index=False, header=False)
        coding_matrix = (coding_matrix * 2 - 1).T
        return coding_matrix, tr_pos_idx, tr_neg_idx

    def create_base_models(self, tr_data, tr_pos_idx, tr_neg_idx):
        models = []
        self.complexity=[]
        for i in range(self.codingLength):
            pos_inst = tr_data[tr_pos_idx[i]]
            neg_inst = tr_data[tr_neg_idx[i]]
            tr_inst = np.vstack((pos_inst, neg_inst))
            tr_labels = np.hstack((np.ones(len(pos_inst)), -np.ones(len(neg_inst))))
            temp=getDataComplexitybyCol(tr_inst,tr_labels)
            self.complexity.append(temp)
            # model = self.estimator().fit(tr_inst, tr_labels)
            # libsvm 使用的训练方式
            prob = svm_problem(tr_labels.tolist(), tr_inst.tolist())
            param = svm_parameter(self.params.get('svm_param'))
            model = svm_train(prob, param)
            models.append(model)
        return models

    def create_base_models_no_complexity(self, tr_data, tr_pos_idx, tr_neg_idx):
        models = []
        for i in range(self.codingLength):
            pos_inst = tr_data[tr_pos_idx[i]]
            neg_inst = tr_data[tr_neg_idx[i]]
            tr_inst = np.vstack((pos_inst, neg_inst))
            tr_labels = np.hstack((np.ones(len(pos_inst)), -np.ones(len(neg_inst))))
            # temp=getDataComplexitybyCol(tr_inst,tr_labels)
            # self.complexity.append(temp)
            # model = self.estimator().fit(tr_inst, tr_labels)
            # libsvm 使用的训练方式
            prob = svm_problem(tr_labels.tolist(), tr_inst.tolist())
            param = svm_parameter(self.params.get('svm_param'))
            model = svm_train(prob, param)
            models.append(model)
        return models

    def create_performance_matrix(self, tr_data, tr_labels):
        performance_matrix = np.zeros((self.num_class, self.codingLength))
        for i in range(self.codingLength):
            model = self.models[i]
            # p_labels = model.predict(tr_data)
            test_label_vector = np.ones(tr_data.shape[0])
            p_labels, _, _ = svm_predict(test_label_vector, tr_data.tolist(), model)
            p_labels = [int(i) for i in p_labels]
            for j in range(self.num_class):
                label_class_j = np.array(p_labels)[tr_labels[j, :] == 1]
                performance_matrix[j, i] = np.abs(sum(label_class_j[label_class_j ==
                                         self.coding_matrix[j, i]])/label_class_j.shape[0])
        return performance_matrix / np.transpose(np.tile(performance_matrix.sum(axis=1), (performance_matrix.shape[1], 1)))

    def fit(self, tr_data, tr_labels):
        self.coding_matrix, tr_pos_idx, tr_neg_idx = self.create_coding_matrix(tr_data, tr_labels)
        self.tr_pos_idx=tr_pos_idx
        self.tr_neg_idx=tr_neg_idx
        self.models = self.create_base_models(tr_data, tr_pos_idx, tr_neg_idx)
        self.performance_matrix = self.create_performance_matrix(tr_data, tr_labels)
        print(self.performance_matrix.shape)

    def predict(self, ts_data, ts_labels):
        bin_pre = None
        decision_pre = None
        for i in range(self.codingLength):
            model = self.models[i]
            test_label_vector = np.ones(ts_data.shape[0])
            p_labels, _, p_vals = svm_predict(test_label_vector, ts_data.tolist(), model)
            # p_labels = model.predict(ts_data)
            # p_vals = model.decision_function(ts_data)
            # p_vals = model.score(ts_data)

            bin_pre = p_labels if bin_pre is None else np.vstack((bin_pre, p_labels))
            decision_pre = np.array(p_vals).T if decision_pre is None else np.vstack((decision_pre, np.array(p_vals).T))

        output_value = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            bin_pre_tmp = bin_pre[:, i]
            decision_pre_tmp = decision_pre[:, i]
            for j in range(self.num_class):
                code = self.coding_matrix[j, :]
                common = np.int8(bin_pre_tmp == code) * self.performance_matrix[j, :] / np.exp(np.abs(decision_pre_tmp))
                error = np.int8(bin_pre_tmp != code) * self.performance_matrix[j, :] * np.exp(np.abs(decision_pre_tmp))
                output_value[j, i] = -sum(common)-sum(error)

        pre_label_matrix = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            idx = output_value[:, i] == max(output_value[:, i])
            pre_label_matrix[idx, i] = 1

        count = 0
        for i in range(ts_data.shape[0]):
            max_idx1 = np.argmax(pre_label_matrix[:, i])
            max_idx2 = np.argmax(ts_labels[:, i])
            if max_idx1 == max_idx2:
                count = count+1
        accuracy = count / ts_data.shape[0]
        return pre_label_matrix, accuracy

    def refit_predict(self,tr_data,tr_labels, ts_data, ts_labels,pre_accuracy):
        coding_matrix=self.coding_matrix
        codingLength=self.codingLength
        self.accuracyList=[]
        # 测试前10列
        for i in range(10):
            tr_pos_idx=self.tr_pos_idx
            tr_neg_idx=self.tr_neg_idx
            self.coding_matrix=coding_matrix
            #移除列
            tr_pos_idx.remove(tr_pos_idx[i])
            tr_neg_idx.remove(tr_neg_idx[i])
            temp=coding_matrix.transpose().tolist()
            temp.remove(temp[i])
            self.coding_matrix=numpy.array(temp).transpose()
            self.codingLength=self.codingLength-1
            self.models = self.create_base_models_no_complexity(tr_data, tr_pos_idx, tr_neg_idx)
            self.performance_matrix = self.create_performance_matrix(tr_data, tr_labels)
            pre_label_matrix, accuracy=self.predict(tr_data,tr_labels)
            self.accuracyList.append(accuracy)
        
        # 比较前10列
        posCol=[]
        negCol=[]
        for i in range(1):
            if self.accuracyList[i]-pre_accuracy>=0:
                posCol.append(i)
            else:
                negCol.append(i)
        print("积极列：")
        for item in posCol:
            print(str(self.accuracyList[item]-pre_accuracy)+" "+self.complexity[item])
        print("消极列：")
        for item in negCol:
            print(str(self.accuracyList[item]-pre_accuracy)+" "+self.complexity[item])
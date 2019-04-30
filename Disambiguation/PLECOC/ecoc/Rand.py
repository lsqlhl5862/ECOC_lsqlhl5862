import numpy as np
from ecoc.BasePLECOC import BasePLECOC
from sklearn.svm import libsvm
from svmutil import *
from GetComplexity import *
from sklearn import preprocessing
from ecoc.PreKNN import PreKNN, PLFeatureSelection
from CodeMatrix.Matrix_tool import _exist_same_col, _exist_same_row, _exist_two_class


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
        self.fs_models = []
        self.plfs=None

    def create_integrity_coding_matrix(self, tr_data, tr_labels):
        num_tr = tr_data.shape[0]
        self.num_class = tr_labels.shape[0]
        self.codingLength = int(np.ceil(10 * np.log2(self.num_class)))
        self.min_num_tr = int(np.ceil(0.1 * num_tr))

        coding_matrix = None
        final_coding_matrix=None
        counter = 0
        tmp_counter=0
        tmp_counter_iter=int(np.ceil(self.codingLength/4))
        tr_pos_idx = []
        tr_neg_idx = []
        final_tr_pos_idx=[]
        final_tr_neg_idx=[]
        tr_data_flag=np.zeros(num_tr)

        # test code start
        # csv_path = 'csv/matrix_dump.csv'
        # if os.path.exists(csv_path):
        #     os.remove(csv_path)
        # test_code_matrix = pd.read_csv(csv_path, header=-1).values
        # for i in range(test_code_matrix.shape[1]):
        # test code end
        for i in range(self.max_iter):
            tmpcode = np.int8(np.random.rand(self.num_class) > 0.5)
            if final_coding_matrix is not None:
                tmp_code_matrix = np.vstack((final_coding_matrix, tmpcode))
                while _exist_same_row(tmp_code_matrix):
                    tmpcode = np.int8(np.random.rand(self.num_class) > 0.5)
                    tmp_code_matrix = np.vstack((final_coding_matrix, tmpcode))
            # tmpcode = test_code_matrix[:, i]
            tmp_pos_idx = []
            tmp_neg_idx = []
            tmp_tr_data_flag=tr_data_flag.copy()
            for j in range(num_tr):
                if np.all((tr_labels[:, j] & tmpcode) == tr_labels[:, j]):
                    tmp_pos_idx.append(j)
                    tmp_tr_data_flag[j]+=1
                else:
                    if np.all((tr_labels[:, j] & np.int8(np.logical_not(tmpcode))) == tr_labels[:, j]):
                        tmp_neg_idx.append(j)
                        tmp_tr_data_flag[j]+=1
            num_pos = len(tmp_pos_idx)
            num_neg = len(tmp_neg_idx)

            if (num_pos+num_neg >= self.min_num_tr) and (num_pos >= 5) and (num_neg >= 5) and (len(np.where(tmp_tr_data_flag==0)[0])==0 or len(np.where(tmp_tr_data_flag==0)[0])<len(np.where(tr_data_flag==0)[0])):
                tmp_counter = tmp_counter + 1
                tr_pos_idx.append(tmp_pos_idx)
                tr_neg_idx.append(tmp_neg_idx)
                coding_matrix = tmpcode if coding_matrix is None else np.vstack(
                    (coding_matrix, tmpcode))
                # tr_data_flag=tmp_tr_data_flag

            if tmp_counter == tmp_counter_iter:
                tmp_counter=0
                high_score_list=self.plfs.matrix_test(coding_matrix,tr_pos_idx,tr_neg_idx)
                if self.codingLength-counter<len(high_score_list):
                    high_score_list=high_score_list[:self.codingLength-counter]
                counter+=len(high_score_list)
                final_coding_matrix= coding_matrix[high_score_list,:] if final_coding_matrix is None else np.vstack(
                    (final_coding_matrix,coding_matrix[high_score_list,:]))
                coding_matrix=None
                for item in high_score_list:
                    final_tr_pos_idx.append(tr_pos_idx[item])
                    final_tr_neg_idx.append(tr_neg_idx[item])
                    for index in tr_pos_idx[item]:
                        tr_data_flag[index]+=1
                    for index in tr_neg_idx[item]:
                        tr_data_flag[index]+=1
                tr_pos_idx=[]
                tr_neg_idx=[]
        
            if counter >= self.codingLength:
                self.codingLength=counter
                break
        if counter != self.codingLength:
            raise ValueError(
                'The required codeword length %s not satisfied', str(self.codingLength))
            self.codingLength = counter
            if counter == 0:
                raise ValueError('Empty coding matrix')
        # dump_matrix = pd.DataFrame(coding_matrix.T)
        # dump_matrix.to_csv(csv_path, index=False, header=False)
        final_coding_matrix = (final_coding_matrix * 2 - 1).T
        print(len(np.where(tr_data_flag==0)[0]))
        return final_coding_matrix, final_tr_pos_idx, final_tr_neg_idx

    def create_coding_matrix(self, tr_data, tr_labels):
        num_tr = tr_data.shape[0]
        self.num_class = tr_labels.shape[0]
        self.codingLength = int(np.ceil(10 * np.log2(self.num_class)))
        self.min_num_tr = int(np.ceil(0.1 * num_tr))

        coding_matrix = None
        counter = 0
        tr_pos_idx = []
        tr_neg_idx = []
        # complexityList=[]

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
                coding_matrix = tmpcode if coding_matrix is None else np.vstack(
                    (coding_matrix, tmpcode))
                # # 复杂度统计
                # pos_inst = tr_data[tmp_pos_idx]
                # neg_inst = tr_data[tmp_neg_idx]
                # temp_tr_inst = np.vstack((pos_inst, neg_inst))
                # temp_tr_labels = np.hstack(
                #     (np.ones(len(pos_inst)), -np.ones(len(neg_inst))))
                # temp = getDataComplexitybyCol(temp_tr_inst, temp_tr_labels)
                # self.complexityList.append(temp)
            
            if counter == self.codingLength:
                break

        if counter != self.codingLength:
            raise ValueError(
                'The required codeword length %s not satisfied', str(self.codingLength))
            self.codingLength = counter
            if counter == 0:
                raise ValueError('Empty coding matrix')
        # dump_matrix = pd.DataFrame(coding_matrix.T)
        # dump_matrix.to_csv(csv_path, index=False, header=False)
        coding_matrix = (coding_matrix * 2 - 1).T
        return coding_matrix, tr_pos_idx, tr_neg_idx

    def create_fs_base_models(self, tr_data, tr_pos_idx, tr_neg_idx, num_feature,tv_data,tv_labels):
        models = []
        # self.complexity=[]
        for i in range(self.codingLength):
            pos_inst = tr_data[tr_pos_idx[i]]
            neg_inst = tr_data[tr_neg_idx[i]]
            tr_inst = np.vstack((pos_inst, neg_inst))
            tr_labels = np.hstack(
                (np.ones(len(pos_inst)), -np.ones(len(neg_inst))))
            # temp=getDataComplexitybyCol(tr_inst,tr_labels)
            # self.complexity.append(temp)
            # model = self.estimator().fit(tr_inst, tr_labels)

            # 使用PLFS
            # plfs = PLFeatureSelection(tr_data,tv_data,tv_labels)
            fs_model=self.plfs.fit(tr_inst, tr_labels,self.coding_matrix[:,i])
            self.fs_models.append(fs_model)
            # libsvm 使用的训练方式
            prob = svm_problem(tr_labels.tolist(),
                               fs_model.transform(tr_inst).tolist())
            param = svm_parameter(self.params.get('svm_param'))
            model = svm_train(prob, param)
            models.append(model)
        return models

    def create_base_models(self, tr_data, tr_pos_idx, tr_neg_idx):
        models = []
        # self.complexity=[]
        for i in range(self.codingLength):
            pos_inst = tr_data[tr_pos_idx[i]]
            neg_inst = tr_data[tr_neg_idx[i]]
            tr_inst = np.vstack((pos_inst, neg_inst))
            tr_labels = np.hstack(
                (np.ones(len(pos_inst)), -np.ones(len(neg_inst))))
            # temp=getDataComplexitybyCol(tr_inst,tr_labels)
            # self.complexity.append(temp)
            # model = self.estimator().fit(tr_inst, tr_labels)

            # # 使用PLFS
            # plfs = PLFeatureSelection(num_feature)
            # plfs.fit(tr_inst, tr_labels,tv_data,tv_labels)
            # self.fs_models.append(plfs)
            # libsvm 使用的训练方式
            prob = svm_problem(tr_labels.tolist(),
                               tr_inst.tolist())
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
            p_labels, _, _ = svm_predict(
                test_label_vector, self.fs_models[i].transform(tr_data).tolist(), model)
            p_labels = [int(i) for i in p_labels]
            for j in range(self.num_class):
                label_class_j = np.array(p_labels)[tr_labels[j, :] == 1]
                performance_matrix[j, i] = np.abs(sum(label_class_j[label_class_j ==
                                                                    self.coding_matrix[j, i]])/label_class_j.shape[0])
        return performance_matrix / np.transpose(np.tile(performance_matrix.sum(axis=1), (performance_matrix.shape[1], 1)))

    def create_base_performance_matrix(self, tr_data, tr_labels):
        performance_matrix = np.zeros((self.num_class, self.codingLength))
        for i in range(self.codingLength):
            model = self.base_models[i]
            # p_labels = model.predict(tr_data)
            test_label_vector = np.ones(tr_data.shape[0])
            p_labels, _, _ = svm_predict(
                test_label_vector, tr_data.tolist(), model)
            p_labels = [int(i) for i in p_labels]
            for j in range(self.num_class):
                label_class_j = np.array(p_labels)[tr_labels[j, :] == 1]
                performance_matrix[j, i] = np.abs(sum(label_class_j[label_class_j ==
                                                                    self.coding_matrix[j, i]])/label_class_j.shape[0])
        return performance_matrix / np.transpose(np.tile(performance_matrix.sum(axis=1), (performance_matrix.shape[1], 1)))

    def fit(self, tr_data, tr_labels):
        self.coding_matrix, tr_pos_idx, tr_neg_idx = self.create_coding_matrix(
            tr_data, tr_labels)
        self.tr_pos_idx = tr_pos_idx
        self.tr_neg_idx = tr_neg_idx
        self.models = self.create_base_models(
            tr_data, tr_pos_idx, tr_neg_idx, tr_data.shape[1])
        self.performance_matrix = self.create_performance_matrix(
            tr_data, tr_labels)
        print(self.performance_matrix.shape)

    def fit_predict(self, tr_data, tr_labels, ts_data, ts_labels,tv_data,tv_labels, pre_knn):
        self.plfs = PLFeatureSelection(tr_data,tr_labels,tv_data,tv_labels,self.params)
        self.coding_matrix, tr_pos_idx, tr_neg_idx = self.create_integrity_coding_matrix(
            tr_data, tr_labels)
        # self.coding_matrix, tr_pos_idx, tr_neg_idx = self.create_coding_matrix(
        #     tr_data, tr_labels)
        self.tr_pos_idx = tr_pos_idx
        self.tr_neg_idx = tr_neg_idx
        # repeat=int(tr_data.shape[1]/3)
        # if(repeat>15):
        #     repeat=15
        temp = []
        self.base_models = self.create_base_models(
            tr_data, tr_pos_idx, tr_neg_idx)
        self.models = self.create_fs_base_models(
            tr_data, tr_pos_idx, tr_neg_idx, tr_data.shape[1],tv_data,tv_labels)
        self.base_performance_matrix = self.create_performance_matrix(
            tr_data, tr_labels)
        self.performance_matrix = self.create_performance_matrix(
            tr_data, tr_labels)
        print(self.performance_matrix.shape)
        matrix, base_accuracy,base_com_1_accuracy = self.base_predict(
            ts_data, ts_labels,pre_knn)
        matrix, base_fs_accuracy, knn_accuracy, com_1_accuracy, com_2_accuracy = self.predict(
            ts_data, ts_labels, pre_knn)
        temp.append(base_accuracy),
        temp.append(base_com_1_accuracy)
        temp.append(base_fs_accuracy) 
        temp.append(knn_accuracy)
        temp.append(com_1_accuracy)
        temp.append(com_2_accuracy)
        return temp
    def predict(self, ts_data, ts_labels, pre_knn):
        bin_pre = None
        decision_pre = None
        for i in range(self.codingLength):
            model = self.models[i]
            fs_model = self.fs_models[i]
            test_label_vector = np.ones(ts_data.shape[0])
            p_labels, _, p_vals = svm_predict(
                test_label_vector, fs_model.transform(ts_data).tolist(), model)
            # p_labels = model.predict(ts_data)
            # p_vals = model.decision_function(ts_data)
            # p_vals = model.score(ts_data)

            bin_pre = p_labels if bin_pre is None else np.vstack(
                (bin_pre, p_labels))
            decision_pre = np.array(p_vals).T if decision_pre is None else np.vstack(
                (decision_pre, np.array(p_vals).T))

        output_value = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            bin_pre_tmp = bin_pre[:, i]
            decision_pre_tmp = decision_pre[:, i]
            for j in range(self.num_class):
                code = self.coding_matrix[j, :]
                common = np.int8(
                    bin_pre_tmp == code) * self.performance_matrix[j, :] / np.exp(np.abs(decision_pre_tmp))
                error = np.int8(
                    bin_pre_tmp != code) * self.performance_matrix[j, :] * np.exp(np.abs(decision_pre_tmp))
                output_value[j, i] = -sum(common)-sum(error)

        output_value = preprocessing.MinMaxScaler().fit_transform(output_value)
        # count_common=0
        # for i in range(len(common_list)):
        #     if(np.array_equal(common_list[i],temp_common)):
        #         count_common+=1
        # print(count_common)
        # for i in range(ts_data.shape[0]):
        #     bin_pre_tmp = bin_pre[:, i]
        #     decision_pre_tmp = decision_pre[:, i]
        #     for j in range(self.num_class):
        #         code = self.coding_matrix[j, :]
        #         common = np.int8(bin_pre_tmp == code) * self.performance_matrix[j, :] / np.exp(np.abs(decision_pre_tmp))
        #         if(j==pre_labels[i]):
        #             self.data_decode.set_cols_list(np.where(common==0),np.where(common!=0))
        #         error = np.int8(bin_pre_tmp != code) * self.performance_matrix[j, :] * np.exp(np.abs(decision_pre_tmp))
        #         output_value[j, i] = -sum(common)-sum(error)

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
        base_accuracy = count / ts_data.shape[0]

        print(base_accuracy)

        _,knn_accuracy,pre_knn_matrix=pre_knn.predict(ts_data,ts_labels)
        print(knn_accuracy)

        tv_data,tv_labels=pre_knn.getValidationData()
        _,tv_knn_accuracy,knn_matrix=pre_knn.predict(tv_data,tv_labels)
        ecoc_matrix=self.fs_base_predict(tv_data,tv_labels)
        weight=pre_knn.getWeight(knn_matrix,ecoc_matrix)
        output_1_value = output_value*weight+pre_knn_matrix*(1-weight)
        pre_label_matrix = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            idx = output_1_value[:, i] == max(output_1_value[:, i])
            pre_label_matrix[idx, i] = 1

        count = 0
        for i in range(ts_data.shape[0]):
            max_idx1 = np.argmax(pre_label_matrix[:, i])
            max_idx2 = np.argmax(ts_labels[:, i])
            if max_idx1 == max_idx2:
                count = count+1
        com_1_accuracy = count / ts_data.shape[0]
        print(com_1_accuracy)

        output_2_value = pre_knn_matrix*output_value
        pre_label_matrix = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            idx = output_2_value[:, i] == max(output_2_value[:, i])
            pre_label_matrix[idx, i] = 1

        count = 0
        for i in range(ts_data.shape[0]):
            max_idx1 = np.argmax(pre_label_matrix[:, i])
            max_idx2 = np.argmax(ts_labels[:, i])
            if max_idx1 == max_idx2:
                count = count+1
        com_2_accuracy = count / ts_data.shape[0]
        print(com_2_accuracy)

        # pre_knn_matrix=pre_knn.getPreKnnMatrix()
        # output_value=output_value+pre_knn_matrix*0.5
        # pre_label_matrix = np.zeros((self.num_class, ts_data.shape[0]))
        # for i in range(ts_data.shape[0]):
        #     idx = output_value[:, i] == max(output_value[:, i])
        #     pre_label_matrix[idx, i] = 1

        # count = 0
        # for i in range(ts_data.shape[0]):
        #     max_idx1 = np.argmax(pre_label_matrix[:, i])
        #     max_idx2 = np.argmax(ts_labels[:, i])
        #     if max_idx1 == max_idx2:
        #         count = count+1
        # com_accuracy = count / ts_data.shape[0]
        # print(com_accuracy)

        return pre_label_matrix, round(base_accuracy, 4), round(knn_accuracy, 4), round(com_1_accuracy, 4), round(com_2_accuracy, 4)

    def base_validation_predict(self, ts_data,ts_labels):
        bin_pre = None
        decision_pre = None
        for i in range(self.codingLength):
            model = self.base_models[i]
            test_label_vector = np.ones(ts_data.shape[0])
            p_labels, _, p_vals = svm_predict(
                test_label_vector, ts_data.tolist(), model)
            # p_labels = model.predict(ts_data)
            # p_vals = model.decision_function(ts_data)
            # p_vals = model.score(ts_data)

            bin_pre = p_labels if bin_pre is None else np.vstack(
                (bin_pre, p_labels))
            decision_pre = np.array(p_vals).T if decision_pre is None else np.vstack(
                (decision_pre, np.array(p_vals).T))

        output_value = np.zeros((self.num_class, ts_data.shape[0]))

        for i in range(ts_data.shape[0]):
            bin_pre_tmp = bin_pre[:, i]
            decision_pre_tmp = decision_pre[:, i]
            for j in range(self.num_class):
                code = self.coding_matrix[j, :]
                common = np.int8(
                    bin_pre_tmp == code) * self.base_performance_matrix[j, :] / np.exp(np.abs(decision_pre_tmp))
                error = np.int8(
                    bin_pre_tmp != code) * self.base_performance_matrix[j, :] * np.exp(np.abs(decision_pre_tmp))
                output_value[j, i] = -sum(common)-sum(error)

        output_value = preprocessing.MinMaxScaler().fit_transform(output_value)
        
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
        base_accuracy = count / ts_data.shape[0]

        return pre_label_matrix

    def repredict(self, ts_data):
        bin_pre = None
        decision_pre = None
        for i in range(self.codingLength):
            model = self.models[i]
            test_label_vector = np.ones(ts_data.shape[0])
            p_labels, _, p_vals = svm_predict(
                test_label_vector, ts_data.tolist(), model)
            # p_labels = model.predict(ts_data)
            # p_vals = model.decision_function(ts_data)
            # p_vals = model.score(ts_data)

            bin_pre = p_labels if bin_pre is None else np.vstack(
                (bin_pre, p_labels))
            decision_pre = np.array(p_vals).T if decision_pre is None else np.vstack(
                (decision_pre, np.array(p_vals).T))

        output_value = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            bin_pre_tmp = bin_pre[:, i]
            decision_pre_tmp = decision_pre[:, i]
            for j in range(self.num_class):
                code = self.coding_matrix[j, :]
                common = np.int8(
                    bin_pre_tmp == code) * self.performance_matrix[j, :] / np.exp(np.abs(decision_pre_tmp))
                error = np.int8(
                    bin_pre_tmp != code) * self.performance_matrix[j, :] * np.exp(np.abs(decision_pre_tmp))
                output_value[j, i] = -sum(common)-sum(error)

        pre_label_matrix = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            idx = output_value[:, i] == max(output_value[:, i])
            pre_label_matrix[idx, i] = 1
        return pre_label_matrix

    def refit_predict(self, tr_data, tr_labels, ts_data, ts_labels, pre_accuracy):
        coding_matrix = self.coding_matrix
        codingLength = self.codingLength
        self.accuracyList = []
        # 测试前10列
        for i in range(codingLength):
            tr_pos_idx = self.tr_pos_idx.copy()
            tr_neg_idx = self.tr_neg_idx.copy()
            self.coding_matrix = coding_matrix
            # 移除列
            tr_pos_idx.remove(tr_pos_idx[i])
            tr_neg_idx.remove(tr_neg_idx[i])
            temp = coding_matrix.transpose().tolist()
            temp.remove(temp[i])
            self.coding_matrix = numpy.array(temp).transpose()
            self.codingLength = codingLength-1
            self.models = self.create_base_models_no_complexity(
                tr_data, tr_pos_idx, tr_neg_idx)
            self.performance_matrix = self.create_performance_matrix(
                tr_data, tr_labels)
            pre_label_matrix, accuracy = self.predict(ts_data, ts_labels)
            self.accuracyList.append(accuracy)

        # 比较前10列
        posCol = []
        negCol = []
        for i in range(codingLength):
            if self.accuracyList[i]-pre_accuracy <= 0:
                posCol.append(i)
            else:
                negCol.append(i)
        print("积极列：")
        for item in posCol:
            print(
                str(self.accuracyList[item]-pre_accuracy)+" "+str(self.complexity[item]))
        print("消极列：")
        for item in negCol:
            print(
                str(self.accuracyList[item]-pre_accuracy)+" "+str(self.complexity[item]))

    def reshape(self, times, tr_data, tr_labels):
        coding_matrix = []
        for i in range(times):
            print(tr_labels.shape[0])
            temp_coding_matrix, tr_pos_idx, tr_neg_idx = self.create_coding_matrix(
                tr_data.copy(), tr_labels.copy())
            temp_complexity_list = []
            for i in range(self.codingLength):
                pos_inst = tr_data[tr_pos_idx[i]]
                neg_inst = tr_data[tr_neg_idx[i]]
                tr_inst = np.vstack((pos_inst, neg_inst))
                tr_labels = np.hstack(
                    (np.ones(len(pos_inst)), -np.ones(len(neg_inst))))
                temp_complexity = getDataComplexitybyCol(tr_inst, tr_labels)
                temp_complexity_list.append(temp_complexity)
            print(temp_complexity_list)
            # f1_mean=np.array(temp_complexity_list).mean(axis=1)
            # print(f1_mean)
    def fs_base_predict(self,ts_data,ts_labels):
        bin_pre = None
        decision_pre = None
        for i in range(self.codingLength):
            model = self.models[i]
            fs_model = self.fs_models[i]
            test_label_vector = np.ones(ts_data.shape[0])
            p_labels, _, p_vals = svm_predict(
                test_label_vector, fs_model.transform(ts_data).tolist(), model)
            # p_labels = model.predict(ts_data)
            # p_vals = model.decision_function(ts_data)
            # p_vals = model.score(ts_data)

            bin_pre = p_labels if bin_pre is None else np.vstack(
                (bin_pre, p_labels))
            decision_pre = np.array(p_vals).T if decision_pre is None else np.vstack(
                (decision_pre, np.array(p_vals).T))

        output_value = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            bin_pre_tmp = bin_pre[:, i]
            decision_pre_tmp = decision_pre[:, i]
            for j in range(self.num_class):
                code = self.coding_matrix[j, :]
                common = np.int8(
                    bin_pre_tmp == code) * self.performance_matrix[j, :] / np.exp(np.abs(decision_pre_tmp))
                error = np.int8(
                    bin_pre_tmp != code) * self.performance_matrix[j, :] * np.exp(np.abs(decision_pre_tmp))
                output_value[j, i] = -sum(common)-sum(error)

        output_value = preprocessing.MinMaxScaler().fit_transform(output_value)
        # count_common=0
        # for i in range(len(common_list)):
        #     if(np.array_equal(common_list[i],temp_common)):
        #         count_common+=1
        # print(count_common)
        # for i in range(ts_data.shape[0]):
        #     bin_pre_tmp = bin_pre[:, i]
        #     decision_pre_tmp = decision_pre[:, i]
        #     for j in range(self.num_class):
        #         code = self.coding_matrix[j, :]
        #         common = np.int8(bin_pre_tmp == code) * self.performance_matrix[j, :] / np.exp(np.abs(decision_pre_tmp))
        #         if(j==pre_labels[i]):
        #             self.data_decode.set_cols_list(np.where(common==0),np.where(common!=0))
        #         error = np.int8(bin_pre_tmp != code) * self.performance_matrix[j, :] * np.exp(np.abs(decision_pre_tmp))
        #         output_value[j, i] = -sum(common)-sum(error)

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
        base_accuracy = count / ts_data.shape[0]

        return pre_label_matrix

    def base_predict(self, ts_data,ts_labels,pre_knn):
        bin_pre = None
        decision_pre = None
        for i in range(self.codingLength):
            model = self.base_models[i]
            test_label_vector = np.ones(ts_data.shape[0])
            p_labels, _, p_vals = svm_predict(
                test_label_vector, ts_data.tolist(), model)
            # p_labels = model.predict(ts_data)
            # p_vals = model.decision_function(ts_data)
            # p_vals = model.score(ts_data)

            bin_pre = p_labels if bin_pre is None else np.vstack(
                (bin_pre, p_labels))
            decision_pre = np.array(p_vals).T if decision_pre is None else np.vstack(
                (decision_pre, np.array(p_vals).T))

        output_value = np.zeros((self.num_class, ts_data.shape[0]))

        for i in range(ts_data.shape[0]):
            bin_pre_tmp = bin_pre[:, i]
            decision_pre_tmp = decision_pre[:, i]
            for j in range(self.num_class):
                code = self.coding_matrix[j, :]
                common = np.int8(
                    bin_pre_tmp == code) * self.base_performance_matrix[j, :] / np.exp(np.abs(decision_pre_tmp))
                error = np.int8(
                    bin_pre_tmp != code) * self.base_performance_matrix[j, :] * np.exp(np.abs(decision_pre_tmp))
                output_value[j, i] = -sum(common)-sum(error)

        output_value = preprocessing.MinMaxScaler().fit_transform(output_value)
        
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
        base_accuracy = count / ts_data.shape[0]

        _,_,pre_knn_matrix=pre_knn.predict(ts_data,ts_labels)

        tv_data,tv_labels=pre_knn.getValidationData()
        _,tv_knn_accuracy,knn_matrix=pre_knn.predict(tv_data,tv_labels)
        ecoc_matrix=self.base_validation_predict(tv_data,tv_labels)
        weight=pre_knn.getWeight(knn_matrix,ecoc_matrix)
        output_1_value = output_value*weight+pre_knn_matrix*(1-weight)
        pre_label_matrix = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            idx = output_1_value[:, i] == max(output_1_value[:, i])
            pre_label_matrix[idx, i] = 1

        count = 0
        for i in range(ts_data.shape[0]):
            max_idx1 = np.argmax(pre_label_matrix[:, i])
            max_idx2 = np.argmax(ts_labels[:, i])
            if max_idx1 == max_idx2:
                count = count+1
        com_1_accuracy = count / ts_data.shape[0]
        print(com_1_accuracy)

        return pre_label_matrix,round(base_accuracy, 4),round(com_1_accuracy, 4)

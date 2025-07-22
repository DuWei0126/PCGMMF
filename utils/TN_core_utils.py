import numpy as np
import pandas as pd
import torch
from utils.utils_tt import *
import os
from datasets.dataset_generic_tt import save_splits
from TN_model import TN_TOAD
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, f1_score, recall_score, confusion_matrix, auc, accuracy_score, matthews_corrcoef
from sklearn.metrics import auc as calc_auc
from TN_text_data import get_genes_by_case_id, get_clinicals_by_case_id, max_min, get_dna_methylation_by_case_id
import matplotlib.pyplot as plt
from scipy.stats import beta

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, count, correct,c):
        self.data[c]["count"] += count
        self.data[c]["correct"] += correct
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.train_loss_min = np.Inf

    def __call__(self, epoch, train_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -train_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(train_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(train_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, train_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.train_loss_min:.6f} --> {train_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.train_loss_min = train_loss

def train(datasets ,cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter  # 可视化
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/test splits...', end=' ')
    train_split, test_split = datasets

    # save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Testing on {} samples".format(len(test_split)))

    #print(train_split[0][0],train_split[0][1],train_split[0][2])
    #损失函数
    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.BCELoss()

    print('\nInit Model...', end=' ')

    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    #model = TOAD_mtl(**model_dict)
    model = TN_TOAD(**model_dict)

    model.relocate()
    print('Done!')
    print_network(model)
    print('\nInit optimizer ...', end=' ')
    #梯度下降法
    optimizer = get_optim(model, args)

    print('Done!')
    
    print('\nInit Loaders...', end=' ')

    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    datasize = len(train_split)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience =15, stop_epoch=100, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        print("ecpoh:  ",epoch)
        # print("Lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        train_loss = train_loop(epoch, model, train_loader, optimizer, args.n_classes, datasize, writer, loss_fn)
        stop = validate(cur, epoch,model,train_loss, early_stopping,  loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, train_error, train_auc, _, train_pre, train_f1, train_recall, train_specificity, train_sensitivity, train_mcc, auc_ci_lower, auc_ci_upper, acc_ci_lower, acc_ci_upper, pre_ci_lower, pre_ci_upper, f1_ci_lower, f1_ci_upper, mcc_ci_lower, mcc_ci_upper = summary(model, train_loader, args.n_classes)
    # print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger, test_pre, test_f1, test_recall, test_specificity, test_sensitivity, test_mcc, auc_ci_lower, auc_ci_upper, acc_ci_lower, acc_ci_upper, pre_ci_lower, pre_ci_upper, f1_ci_lower, f1_ci_upper, mcc_ci_lower, mcc_ci_upper  = summary(model, test_loader, args.n_classes)

    print('Test_Error: {:.4f} ,Auc: {:.4f}, Pre: {:.4f}, F1: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, Sensitivity: {:.4f}, Mcc: {:.4f}'.format(test_error, test_auc, test_pre, test_f1, test_recall, test_specificity, test_sensitivity, test_mcc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
    
    writer.close()
    return results_dict, test_auc, 1-test_error, test_pre, test_f1, test_recall, test_specificity, test_sensitivity, test_mcc, auc_ci_lower, auc_ci_upper, acc_ci_lower, acc_ci_upper, pre_ci_lower, pre_ci_upper, f1_ci_lower, f1_ci_upper, mcc_ci_lower, mcc_ci_upper

def train_loop(epoch, model, loader, optimizer, n_classes, datasize, writer = None, loss_fn = None,):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    hat = np.zeros(len(loader))

    # for batch_idx, (data,label,gender...) in enumerate(loader):
    for batch_idx, (data, label, case_id) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        dna_methylation = get_dna_methylation_by_case_id(case_id[0])
        dna_methylation = dna_methylation.values
        dna_methylation_data = torch.tensor(dna_methylation, dtype=torch.float32)
        dna_methylation_data = dna_methylation_data.to(device)

        #mirna
        """mirna = get_mirna_by_case_id(case_id[0])
        mirna = mirna.values
        mirna = torch.tensor(mirna, dtype=torch.float32)
        mirna = mirna.to(device)"""
        #获取当前样本的基因数据
        genes_list = get_genes_by_case_id(case_id[0])
        genes_list = genes_list.values
        gene_data = torch.tensor(genes_list, dtype=torch.float32)
        #gene_data = torch.unsqueeze(gene_data, dim=0)
        gene_data = gene_data.to(device)


        #获取当前样本的临床数据
        age_list = []
        clinicals_list, clinicals_age = get_clinicals_by_case_id(case_id[0])
        # clinical_data
        clinicals_list = clinicals_list.values.tolist()
        clinicals_list = list(clinicals_list[0])

        string_to_int = {'T1': 1, 'T1a': 1, 'T1b': 1, 'T1c': 1, 'T2': 2, 'T2b': 2, 'T3': 3, 'T3a': 3, 'T4': 4,
                         'T4b': 4, 'T4d': 4, 'TX': 5, 'N0': 0, 'N0 (i-)': 0, 'N0 (i+)': 0, 'N0 (mol+)': 0, 'N1': 1,
                         'N1a': 1, 'N1b': 1, 'N1c': 1, 'N1mi': 1, 'N2': 2, 'N2a': 2, 'N3': 3, 'N3a': 3, 'N3b': 3,
                         'N3c': 3, 'NX': 4, 'cM0 (i+)': 0, 'M0': 0, 'M1': 1, 'MX': 2, 'Stage I': 1, 'Stage IA': 1,
                         'Stage IB': 1, 'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2, 'Stage III': 3,
                         'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3, 'Stage IV': 4, 'Stage X': 5, 'positive': 0, 'negative':1}

        """string_to_int = {
            'T1': 1, 'T1a': 1, 'T1b': 1, 'T1c': 1, 'T2': 2, 'T2b': 2, 'T3': 3, 'T3a': 3, 'T4': 4, 'T4b': 4, 'T4d': 4,
            'TX': 10, 'T0': 0,
            'N0': 0, 'N0 (i-)': 0, 'N0 (i+)': 0, 'N0 (mol+)': 0, 'N1': 1, 'N1a': 1, 'N1b': 1, 'N1c': 1, 'N1mi': 1,
            'N2': 2, 'N2a': 2, 'N3': 3, 'N3a': 3, 'N3b': 3, 'N3c': 3, 'NX': 10,
            'cM0 (i+)': 0, 'M0': 0, 'M1': 1, 'MX': 10,
            'Stage I': 1, 'Stage IA': 1,'Stage IB': 1, 'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2, 'Stage III': 3,
            'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3, 'Stage IV': 4, 'Stage X': 10
        }"""


        """string_to_int = {'T1': 0.1, 'T1a': 0.1, 'T1b': 0.1, 'T1c': 0.1, 'T2': 0.2, 'T2b': 0.2, 'T3': 0.3,
                         'T3a': 0.3, 'T4': 0.4, 'T4b': 0.4, 'T4d': 0.4, 'TX': 1, 'N0': 0, 'N0 (i-)': 0,
                         'N0 (i+)': 0, 'N0 (mol+)': 0, 'N1': 0.1, 'N1a': 0.1, 'N1b': 0.1, 'N1c': 0.1, 'N1mi': 0.1,
                         'N2': 0.2, 'N2a': 0.2, 'N3': 0.3, 'N3a': 0.3, 'N3b': 0.3, 'N3c': 0.3, 'NX': 0.4,
                         'cM0 (i+)': 0, 'M0': 0, 'M1': 0.1, 'MX': 0.2, 'Stage I': 0.1, 'Stage IA': 0.1,
                         'Stage IB': 0.1, 'Stage II': 0.2, 'Stage IIA': 0.2, 'Stage IIB': 0.2, 'Stage III': 0.3,
                         'Stage IIIA': 0.3, 'Stage IIIB': 0.3, 'Stage IIIC': 0.3, 'Stage IV': 0.4, 'Stage X': 1}"""

        data_list_numeric = [string_to_int[item] if item in string_to_int else item for item in clinicals_list]
        clinical_data = torch.tensor(data_list_numeric, dtype=torch.float32)
        clinicals_data_tensor = torch.unsqueeze(clinical_data, dim=0)
        clinicals_data_tensor = clinicals_data_tensor.to(device)

        # 临床年龄处理
        clinicals_age = max_min(clinicals_age)
        clinicals_age = np.array(clinicals_age)
        #age_list.append(clinicals_age)
        clinicals_age_tensor = torch.tensor(clinicals_age, dtype=torch.float32)
        clinicals_age_tensor = torch.unsqueeze(clinicals_age_tensor, dim=0)
        clinicals_age_tensor = clinicals_age_tensor.to(device)


        # label = label.float().to(device)
        # gender = gender.to(device)
        #print(batch_idx, data, label)
        #optimizer.zero_grad()
        optimizer.zero_grad()
        # print(batch_idx, data, label)
        results_dict = model(data,gene_data,clinicals_data_tensor, clinicals_age_tensor, dna_methylation_data)
        #results_dict = model(data)
        #print(batch_idx, results_dict)

        logits, Y_prob, Y_hat= results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']

        #_, preds = torch.max(results_dict, 1)
        prob[batch_idx] = Y_prob.cpu().detach().numpy()

        hat[batch_idx] = Y_hat.cpu().detach().numpy()

        labels[batch_idx] = label.item()
        # print("label and Y_prob ------------------->",label, Y_prob, "\n")
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        #loss = loss_fn(results_dict, label)
        #loss.backward()
        #optimizer.step()
        #optimizer.zero_grad()


        #running_loss += loss.item() * data.size(0)
        #running_corrects += torch.sum((preds == data.data).int())
        
    #epoch_loss = running_loss / datasize
    #epoch_acc = running_corrects / datasize
    
    #print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))   
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error

        loss.backward()
        optimizer.step()

        # backward pass
        #loss.backward()
        #for name, parms in model.named_parameters():
        #    print("grad_value --->",parms.grad)
        # step
        #optimizer.step()


    # print(prob,labels)
    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    # 获取混淆矩阵
    tn, fp, fn, tp = confusion_matrix(labels, hat).ravel()

    # 计算特异性
    specificity = tn / (tn + fp)

    # 计算敏感性
    sensitivity = tp / (tp + fn)

    # 计算马修斯相关系数（MCC）
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    acc = accuracy_score(labels, hat)
    # 计算precision(查准率)
    pre = precision_score(labels, hat)

    # 计算F1分数
    f1 = f1_score(labels, hat)

    # 计算Recall(召回率)
    recall = recall_score(labels, hat)

    # 计算AUC
    auc_value = roc_auc_score(labels, prob[:, 1])

    ci_lower, ci_upper = calculate_auc_ci(auc_value, labels, prob[:, 1])
    acc_ci_lower, acc_ci_upper = calculate_acc_ci(acc, labels, hat)
    pre_ci_lower, pre_ci_upper = calculate_pre_ci(pre, labels, hat)
    f1_ci_lower, f1_ci_upper = calculate_f1_ci(f1, labels, hat)
    mcc_ci_lower, mcc_ci_upper = calculate_mcc_ci(mcc, labels, hat)
    print(f"AUC:95% 置信区间: [{ci_lower}, {ci_upper}]")
    print(f"ACC:95% 置信区间: [{acc_ci_lower}, {acc_ci_upper}]")
    print(f"PRE:95% 置信区间: [{pre_ci_lower}, {pre_ci_upper}]")
    print(f"F1:95% 置信区间: [{f1_ci_lower}, {f1_ci_upper}]")
    print(f"MCC:95% 置信区间: [{mcc_ci_lower}, {mcc_ci_upper}]")


    print('Epoch: {}, Train_loss: {:.4f}, Train_error: {:.4f}, Auc: {:.4f}, Pre: {:.4f}, F1: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, Sensitivity: {:.4f}, Mcc: {:.4f}'.format(epoch, train_loss, train_error, auc_value, pre, f1, recall, specificity, sensitivity, mcc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

    return train_loss



   
def validate(cur, epoch, model,train_loss, early_stopping = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    if early_stopping:
        assert results_dir
        early_stopping(epoch, train_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_hats = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    #for batch_idx, (data, label,gender...) in enumerate(loader):
    for batch_idx, (data, label, case_id) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        # gender = gender.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            #results_dict = model(data,gender...)

            dna_methylation = get_dna_methylation_by_case_id(case_id[0])
            dna_methylation = dna_methylation.values
            dna_methylation_data = torch.tensor(dna_methylation, dtype=torch.float32)
            dna_methylation_data = dna_methylation_data.to(device)


            """# mirna
            mirna = get_mirna_by_case_id(case_id[0])
            mirna = mirna.values
            mirna = torch.tensor(mirna, dtype=torch.float32)
            mirna = mirna.to(device)"""


            # 获取当前样本的基因数据
            genes_list = get_genes_by_case_id(case_id[0])
            genes_list = genes_list.values
            gene_data = torch.tensor(genes_list, dtype=torch.float32)
            # gene_data = torch.unsqueeze(gene_data, dim=0)
            gene_data = gene_data.to(device)


            # 获取当前样本的临床数据
            age_list = []
            clinicals_list, clinicals_age = get_clinicals_by_case_id(case_id[0])
            # clinical_data
            clinicals_list = clinicals_list.values.tolist()
            clinicals_list = list(clinicals_list[0])

            string_to_int = {'T1': 1, 'T1a': 1, 'T1b': 1, 'T1c': 1, 'T2': 2, 'T2b': 2, 'T3': 3, 'T3a': 3, 'T4': 4,
                             'T4b': 4, 'T4d': 4, 'TX': 5, 'N0': 0, 'N0 (i-)': 0, 'N0 (i+)': 0, 'N0 (mol+)': 0, 'N1': 1,
                             'N1a': 1, 'N1b': 1, 'N1c': 1, 'N1mi': 1, 'N2': 2, 'N2a': 2, 'N3': 3, 'N3a': 3, 'N3b': 3,
                             'N3c': 3, 'NX': 4, 'cM0 (i+)': 0, 'M0': 0, 'M1': 1, 'MX': 2, 'Stage I': 1, 'Stage IA': 1,
                             'Stage IB': 1, 'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2, 'Stage III': 3,
                             'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3, 'Stage IV': 4, 'Stage X': 5,
                             'positive': 0, 'negative': 1}

            data_list_numeric = [string_to_int[item] if item in string_to_int else item for item in clinicals_list]
            clinical_data = torch.tensor(data_list_numeric, dtype=torch.float32)
            clinicals_data_tensor = torch.unsqueeze(clinical_data, dim=0)
            clinicals_data_tensor = clinicals_data_tensor.to(device)
            # 临床年龄处理
            clinicals_age = max_min(clinicals_age)
            clinicals_age = np.array(clinicals_age)
            # age_list.append(clinicals_age)
            clinicals_age_tensor = torch.tensor(clinicals_age, dtype=torch.float32)
            clinicals_age_tensor = torch.unsqueeze(clinicals_age_tensor, dim=0)
            clinicals_age_tensor = clinicals_age_tensor.to(device)

            #results_dict = model(data)
            results_dict = model(data,gene_data,clinicals_data_tensor, clinicals_age_tensor, dna_methylation_data)
            logits, Y_prob, Y_hat = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']
            del results_dict

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        hats = Y_hat.cpu().numpy()
        all_probs[batch_idx] = probs
        all_hats[batch_idx] = hats
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        print("all_labels", all_labels)
        print("all_probs[:, 1]", all_probs[:, 1])
        auc_value = roc_auc_score(all_labels, all_probs[:, 1])
        # 获取混淆矩阵
        tn, fp, fn, tp = confusion_matrix(all_labels, all_hats).ravel()

        # 计算特异性
        specificity = tn / (tn + fp)

        # 计算敏感性
        sensitivity = tp / (tp + fn)

        # 计算马修斯相关系数（MCC）
        mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

        acc = accuracy_score(all_labels, all_hats)

        # 计算precision(查准率)
        pre = precision_score(all_labels, all_hats)

        # 计算F1分数
        f1 = f1_score(all_labels, all_hats)

        # 计算Recall(召回率)
        recall = recall_score(all_labels, all_hats)

        fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])
        roc_auc_value = auc(fpr, tpr)


        auc_ci_lower, auc_ci_upper = calculate_auc_ci(auc_value, all_labels, all_probs[:, 1])
        acc_ci_lower, acc_ci_upper = calculate_acc_ci(acc, all_labels, all_hats)
        pre_ci_lower, pre_ci_upper = calculate_pre_ci(pre, all_labels, all_hats)
        f1_ci_lower, f1_ci_upper = calculate_f1_ci(f1, all_labels, all_hats)
        mcc_ci_lower, mcc_ci_upper = calculate_mcc_ci(mcc, all_labels, all_hats)
        print(f"AUC:95% 置信区间: [{auc_ci_lower}, {auc_ci_upper}]")
        print(f"ACC:95% 置信区间: [{acc_ci_lower}, {acc_ci_upper}]")
        print(f"PRE:95% 置信区间: [{pre_ci_lower}, {pre_ci_upper}]")
        print(f"F1:95% 置信区间: [{f1_ci_lower}, {f1_ci_upper}]")
        print(f"MCC:95% 置信区间: [{mcc_ci_lower}, {mcc_ci_upper}]")
        # 绘制 ROC 曲线

        """plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_value)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")"""

        # 保存图像
        #plt.savefig('roc_curve.png')


    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

    return patient_results, test_error, auc_value, acc_logger, pre, f1, recall, specificity, sensitivity, mcc, auc_ci_lower, auc_ci_upper, acc_ci_lower, acc_ci_upper, pre_ci_lower, pre_ci_upper, f1_ci_lower, f1_ci_upper, mcc_ci_lower, mcc_ci_upper


def calculate_auc_ci(auc_value, y_true, y_pred, n_bootstrap=1000):
    """
    使用 Bootstrap 方法计算 AUC 的 95% 置信区间

    参数:
    auc_value: 已计算好的 AUC 值
    y_true: 真实标签数组
    y_pred: 预测概率数组
    n_bootstrap: Bootstrap 抽样的次数，默认为 1000

    返回:
    ci_lower: 95% 置信区间的下限
    ci_upper: 95% 置信区间的上限
    """
    auc_values = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred[indices]
        auc_bootstrap = roc_auc_score(y_true_bootstrap, y_pred_bootstrap)
        auc_values.append(auc_bootstrap)

    ci_lower = np.percentile(auc_values, 2.5)
    ci_upper = np.percentile(auc_values, 97.5)

    return ci_lower, ci_upper


def calculate_acc_ci(acc_value, y_true, y_pred, n_bootstrap=1000):
    """
    使用 Bootstrap 方法计算 ACC 的 95% 置信区间

    参数:
    acc_value: 已计算好的 ACC 值
    y_true: 真实标签数组
    y_pred: 预测标签数组
    n_bootstrap: Bootstrap 抽样的次数，默认为 1000

    返回:
    ci_lower: 95% 置信区间的下限
    ci_upper: 95% 置信区间的上限
    """
    acc_values = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred[indices]
        acc_bootstrap = accuracy_score(y_true_bootstrap, y_pred_bootstrap)
        acc_values.append(acc_bootstrap)

    ci_lower = np.percentile(acc_values, 2.5)
    ci_upper = np.percentile(acc_values, 97.5)

    return ci_lower, ci_upper


def calculate_pre_ci(pre_value, y_true, y_pred, n_bootstrap=1000):
    """
    使用 Bootstrap 方法计算 PRE 的 95% 置信区间

    参数:
    pre_value: 已计算好的 PRE 值
    y_true: 真实标签数组
    y_pred: 预测标签数组
    n_bootstrap: Bootstrap 抽样的次数，默认为 1000

    返回:
    ci_lower: 95% 置信区间的下限
    ci_upper: 95% 置信区间的上限
    """
    pre_values = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred[indices]
        pre_bootstrap = precision_score(y_true_bootstrap, y_pred_bootstrap)
        pre_values.append(pre_bootstrap)

    ci_lower = np.percentile(pre_values, 2.5)
    ci_upper = np.percentile(pre_values, 97.5)

    return ci_lower, ci_upper


def calculate_f1_ci(f1_value, y_true, y_pred, n_bootstrap=1000):
    """
    使用 Bootstrap 方法计算 F1-score 的 95% 置信区间

    参数:
    f1_value: 已计算好的 F1-score 值
    y_true: 真实标签数组
    y_pred: 预测标签数组
    n_bootstrap: Bootstrap 抽样的次数，默认为 1000

    返回:
    ci_lower: 95% 置信区间的下限
    ci_upper: 95% 置信区间的上限
    """
    f1_values = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred[indices]
        f1_bootstrap = f1_score(y_true_bootstrap, y_pred_bootstrap)
        f1_values.append(f1_bootstrap)

    ci_lower = np.percentile(f1_values, 2.5)
    ci_upper = np.percentile(f1_values, 97.5)

    return ci_lower, ci_upper


def calculate_mcc_ci(mcc_value, y_true, y_pred, n_bootstrap=1000):
    """
    使用 Bootstrap 方法计算 MCC 的 95% 置信区间

    参数:
    mcc_value: 已计算好的 MCC 值
    y_true: 真实标签数组
    y_pred: 预测标签数组
    n_bootstrap: Bootstrap 抽样的次数，默认为 1000

    返回:
    ci_lower: 95% 置信区间的下限
    ci_upper: 95% 置信区间的上限
    """
    mcc_values = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred[indices]
        mcc_bootstrap = matthews_corrcoef(y_true_bootstrap, y_pred_bootstrap)
        mcc_values.append(mcc_bootstrap)

    ci_lower = np.percentile(mcc_values, 2.5)
    ci_upper = np.percentile(mcc_values, 97.5)

    return ci_lower, ci_upper
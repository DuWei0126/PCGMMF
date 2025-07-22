from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, \
    matthews_corrcoef
from TN_model import TN_TOAD  # 确保 TN_TOAD 模型定义在 TN_model 文件中
from TN_text_data import get_genes_by_case_id, get_clinicals_by_case_id, max_min, get_dna_methylation_by_case_id
import os
import numpy as np
from utils.utils_tt import *
from datasets.dataset_generic_tt import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
import torch
from utils.TN_core_utils import calculate_auc_ci, calculate_acc_ci, calculate_pre_ci, calculate_f1_ci, calculate_mcc_ci


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count

def test_model(model, test_loader, n_classes, device=None):
    """
    测试模型的性能
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(test_loader), n_classes))
    all_labels = np.zeros(len(test_loader))
    all_hats = np.zeros(len(test_loader))

    slide_ids = test_loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label, case_id) in enumerate(test_loader):
        data, label = data.to(device), label.to(device)

        dna_methylation = get_dna_methylation_by_case_id(case_id[0])
        dna_methylation = dna_methylation.values
        dna_methylation_data = torch.tensor(dna_methylation, dtype=torch.float32)
        dna_methylation_data = dna_methylation_data.to(device)

        genes_list = get_genes_by_case_id(case_id[0])
        genes_list = genes_list.values
        gene_data = torch.tensor(genes_list, dtype=torch.float32)
        gene_data = gene_data.to(device)

        clinicals_list, clinicals_age = get_clinicals_by_case_id(case_id[0])
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

        clinicals_age = max_min(clinicals_age)
        clinicals_age = np.array(clinicals_age)
        clinicals_age_tensor = torch.tensor(clinicals_age, dtype=torch.float32)
        clinicals_age_tensor = torch.unsqueeze(clinicals_age_tensor, dim=0)
        clinicals_age_tensor = clinicals_age_tensor.to(device)

        with torch.no_grad():
            results_dict = model(data, gene_data, clinicals_data_tensor, clinicals_age_tensor, dna_methylation_data)
            logits, Y_prob, Y_hat = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']

        probs = Y_prob.cpu().numpy()
        hats = Y_hat.cpu().numpy()
        all_probs[batch_idx] = probs
        all_hats[batch_idx] = hats
        all_labels[batch_idx] = label.item()

        patient_results.update({slide_ids.iloc[batch_idx]: {'slide_id': np.array(slide_ids.iloc[batch_idx]),
                                                            'prob': probs, 'label': label.item()}})
        error = (hats != label.item())
        test_error += error

    test_error /= len(test_loader)

    if n_classes == 2:
        auc_value = roc_auc_score(all_labels, all_probs[:, 1])
        tn, fp, fn, tp = confusion_matrix(all_labels, all_hats).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        acc = accuracy_score(all_labels, all_hats)
        pre = precision_score(all_labels, all_hats)
        f1 = f1_score(all_labels, all_hats)
        recall = recall_score(all_labels, all_hats)

        auc_ci_lower, auc_ci_upper = calculate_auc_ci(auc_value, all_labels, all_probs[:, 1])
        acc_ci_lower, acc_ci_upper = calculate_acc_ci(acc, all_labels, all_hats)
        pre_ci_lower, pre_ci_upper = calculate_pre_ci(pre, all_labels, all_hats)
        f1_ci_lower, f1_ci_upper = calculate_f1_ci(f1, all_labels, all_hats)
        mcc_ci_lower, mcc_ci_upper = calculate_mcc_ci(mcc, all_labels, all_hats)

        print(f"AUC: {auc_value:.4f}, 95% CI: [{auc_ci_lower:.4f}, {auc_ci_upper:.4f}]")
        print(f"ACC: {acc:.4f}, 95% CI: [{acc_ci_lower:.4f}, {acc_ci_upper:.4f}]")
        print(f"PRE: {pre:.4f}, 95% CI: [{pre_ci_lower:.4f}, {pre_ci_upper:.4f}]")
        print(f"F1: {f1:.4f}, 95% CI: [{f1_ci_lower:.4f}, {f1_ci_upper:.4f}]")
        print(f"MCC: {mcc:.4f}, 95% CI: [{mcc_ci_lower:.4f}, {mcc_ci_upper:.4f}]")
        print(f"Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}")

    return patient_results, test_error, auc_value, acc, pre, f1, recall, specificity, sensitivity, mcc, auc_ci_lower, auc_ci_upper, acc_ci_lower, acc_ci_upper, pre_ci_lower, pre_ci_upper, f1_ci_lower, f1_ci_upper, mcc_ci_lower, mcc_ci_upper


seed_torch(5)

dataset = Generic_MIL_Dataset(csv_path ='Step_4.csv',
                                  data_dir="./FEATURES_DIRECTORY_VIL_224",
                                  # data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                                  shuffle = False,
                                  seed = 5,
                                  print_info = True,
                                  label_dict = {'0':0, '1':1},
                                  patient_strat=False
                                  )


if __name__ == "__main__":
    dataset.load_from_h5(True)
    _, test_dataset, _, test_index = dataset.return_splits(from_id=False,
                                                           csv_path=f"./splits/Recurrence_metastasis_vs_normal_100_balance/splits_0.csv")

    datasets = test_dataset

    model_dict = {"dropout": True, 'n_classes': 2}
    # 加载模型和测试数据
    model = TN_TOAD(**model_dict)
    model.load_state_dict(torch.load("./results_5/dataset_1_result_s5/s_1_checkpoint.pt"))

    test_split = datasets
    test_loader = get_split_loader(test_split, testing=True)

    # 测试模型
    results = test_model(model, test_loader, n_classes=2)



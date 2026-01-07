import time
import random
import torch
from datapro import CVEdgeDataset
from model import MSCLCMI
import numpy as np
import torch.utils.data.dataloader as DataLoader
from sklearn.model_selection import KFold
from clac_metric import get_metric
import csv
import copy


# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


def get_metrics(score, label):
    y_pre = score
    y_true = label
    # metric = caculate_metrics(y_pre, y_true)
    metric = get_metric(y_true, y_pre)
    return metric


def print_met(list):
    print('AUC ：%.4f ' % (list[0]),
          'AUPR ：%.4f ' % (list[1]),
          'Accuracy ：%.4f ' % (list[2]),
          'f1_score ：%.4f ' % (list[3]),
          'recall ：%.4f ' % (list[4]),
          'precision ：%.4f ' % (list[5]),
          'specificity ：%.4f \n' % (list[6]))


def train_test(train_data, param, state):
    valid_metric = []
    valid_tpr = []
    valid_fpr = []
    valid_recall = []
    valid_precision = []
    train_edges = train_data['train_Edges']
    train_labels = train_data['train_Labels']
    kfolds = param.kfold
    torch.manual_seed(42)

    if state == 'valid':
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)
        train_idx, valid_idx = [], []
        for train_index, valid_index in kf.split(train_edges):
            train_idx.append(train_index)
            valid_idx.append(valid_index)
        for i in range(kfolds):
            model = MSCLCMI(param)  ##*

            optimizer = torch.optim.Adam(model.parameters(), lr=param.lr, weight_decay=0)  ###

            print(f'Fold {i + 1} ')
            # get train set and valid set
            edges_train, edges_valid = train_edges[train_idx[i]], train_edges[valid_idx[i]]
            labels_train, labels_valid = train_labels[train_idx[i]], train_labels[valid_idx[i]]
            trainEdges = CVEdgeDataset(edges_train, labels_train)
            validEdges = CVEdgeDataset(edges_valid, labels_valid)
            trainLoader = DataLoader.DataLoader(trainEdges, batch_size=param.batchSize, shuffle=True, num_workers=0)
            validLoader = DataLoader.DataLoader(validEdges, batch_size=param.batchSize, shuffle=False, num_workers=0)

            print("-----training-----")

            # 早停机制相关变量
            best_val_metric = 0.0  # 用于监控验证集性能（这里使用AUC作为监控指标）
            patience = getattr(param, 'patience', 10)  # 早停耐心值（可由param覆盖）
            patience_counter = 0  # 当前耐心计数器
            best_model_state = copy.deepcopy(model.state_dict())  # 初始化最佳模型权重，避免未定义

            for e in range(param.epoch):
                running_loss = 0.0  ###
                epo_label = []
                epo_score = []
                # print("epoch：", e + 1)
                model.train()
                start = time.time()
                for i, item in enumerate(trainLoader):
                    data, label = item
                    train_data = data
                    true_label = label  ###
                    pre_score, ssl_loss = model(train_data)  ##*
                    train_loss = torch.nn.BCELoss()
                    loss = train_loss(pre_score, true_label) + ssl_loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    running_loss += loss.item()  ###
                    # print(f"After batch {i + 1}: loss= {loss:.3f};", end='\n')  ###
                    batch_score = pre_score.cpu().detach().numpy()
                    epo_score = np.append(epo_score, batch_score)
                    epo_label = np.append(epo_label, label.numpy())
                end = time.time()

                # ======== 每个 epoch 计算并输出训练集评估指标 ========
                try:
                    tpr, fpr, recall_list, precision_list, metric = get_metrics(epo_score, epo_label)
                except Exception as err:
                    print(f"Warning: error in training metric computation: {err}")
                    metric = [0.0] * 7  # 如果报错，防止程序崩溃

                print(f"Epoch [{e + 1}/{param.epoch}] | Train Loss: {running_loss:.4f} | Time: {end - start:.2f}s")
                print("Training Metrics:")
                print_met(metric)

                # ======== 每个 epoch 计算并输出验证集评估指标 ========
                model.eval()
                val_score, val_label = [], []
                with torch.no_grad():
                    for i, item in enumerate(validLoader):
                        data, label = item
                        train_data = data
                        pre_score, loss = model(train_data)  ##*
                        batch_score = pre_score.cpu().detach().numpy()
                        val_score = np.append(val_score, batch_score)
                        val_label = np.append(val_label, label.numpy())

                try:
                    val_tpr, val_fpr, val_recall_list, val_precision_list, val_metric = get_metrics(val_score,
                                                                                                    val_label)
                except Exception as err:
                    print(f"Warning: error in validation metric computation: {err}")
                    val_metric = [0.0] * 7  # 如果报错，防止程序崩溃

                print("Validation Metrics:")
                print_met(val_metric)

                # ======== 早停机制 ========
                current_val_auc = val_metric[0]  # AUC作为监控指标
                if current_val_auc > best_val_metric:
                    best_val_metric = current_val_auc
                    best_model_state = copy.deepcopy(model.state_dict())  # 保存最佳模型权重
                    patience_counter = 0
                    print(f"Validation AUC improved to {current_val_auc:.4f}")
                else:
                    patience_counter += 1
                    print(f"Validation AUC did not improve. Patience: {patience_counter}/{patience}")

                # 检查是否需要早停
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {e + 1}")
                    break

                print("-" * 50)

                # print('Time：%.2f \n' % (end - start))

            # 使用早停机制中保存的最佳模型权重
            model.load_state_dict(best_model_state)

            # 使用最佳模型重新计算验证集指标
            valid_score, valid_label = [], []
            model.eval()
            with torch.no_grad():
                for i, item in enumerate(validLoader):
                    data, label = item
                    train_data = data
                    pre_score, loss = model(train_data)
                    batch_score = pre_score.cpu().detach().numpy()
                    valid_score = np.append(valid_score, batch_score)
                    valid_label = np.append(valid_label, label.numpy())

            tpr, fpr, recall_list, precision_list, metric = get_metrics(valid_score, valid_label)
            print(f"Final validation results for fold {i + 1}:")
            print_met(metric)
            valid_metric.append(metric)
            valid_tpr.append(tpr)
            valid_fpr.append(fpr)
            valid_recall.append(recall_list)
            valid_precision.append(precision_list)

        # 跨折叠汇总与输出（放在所有fold结束后一次性执行）
        formatted_valid_metric = [[round(item, 4) for item in sublist] for sublist in valid_metric]
        for sublist in formatted_valid_metric:
            print(sublist)
        cv_metric = np.mean(valid_metric, axis=0)
        print_met(cv_metric)
        cv_tpr = np.mean(valid_tpr, axis=0)
        cv_fpr = np.mean(valid_fpr, axis=0)
        cv_recall = np.mean(valid_recall, axis=0)
        cv_precision = np.mean(valid_precision, axis=0)

        valid_tpr.append(cv_tpr)
        valid_fpr.append(cv_fpr)
        valid_recall.append(cv_recall)
        valid_precision.append(cv_precision)

        StorFile(valid_tpr, R'D:\YD\MFERL-main\results\5cv\CircBANK\5cv\tpr_all_ours_9589.csv')
        StorFile(valid_fpr, R'D:\YD\MFERL-main\results\5cv\CircBANK\5cv\fpr_all_ours_9589.csv')
        StorFile(valid_recall, R'D:\YD\MFERL-main\results\5cv\CircBANK5cv\recall_all_ours_9589.csv')
        StorFile(valid_precision, R'D:\YD\MFERL-main\results\5cv\CircBANK\5cv\precision_all_ours_9589.csv')

        return kfolds

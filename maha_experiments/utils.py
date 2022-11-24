from sklearn.metrics import precision_recall_curve, auc

def aupr(y_true, y_pred, pos_label=1):

    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=pos_label)

    return auc(recall, precision)
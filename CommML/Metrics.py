def get_confusion(y,y_pred):
    labels = sorted(list(set(y) | set(y_pred)))  
    size = len(labels)
    
    label_index = {label: idx for idx, label in enumerate(labels)}
    
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    
    for true, pred in zip(y, y_pred):
        i = label_index[true]
        j = label_index[pred]
        matrix[i][j] += 1

    return matrix

def confusion_matrix(y, y_pred):
    matrix  = get_confusion(y,y_pred)
    labels = sorted(list(set(y) | set(y_pred)))  
    print(f"Rows: Actual Label , Columns: Predicted Labels")
    print(f"\n\t{labels}")
    for idx, row in enumerate(matrix):
        print(f"{labels[idx]}  {row}")


def get_accuracy(y, y_pred):
    matrix = get_confusion(y, y_pred)
    true_positive , true_negative , false_positive , false_negative = matrix[0][0] , matrix[1][1], matrix[0][1], matrix[1][0]
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    return accuracy

def get_precision(y, y_pred):
    matrix = get_confusion(y, y_pred)
    true_positive , false_positive = matrix[0][0] ,  matrix[0][1]
    precision = (true_positive) / (true_positive + false_positive)
    return precision

def get_recall(y,y_pred):
    matrix = get_confusion(y, y_pred)
    true_positive , false_negative = matrix[0][0] , matrix[1][0]
    recall = (true_positive) / (true_positive + false_negative)
    return recall

def f1_score(y, y_pred):
    precision = get_precision(y,y_pred)
    recall = get_recall(y,y_pred)
    f1 = 2 * ( (precision * recall) / (precision + recall))
    return f1

import torch

__all__ = [
    "get_f1_score",
    "get_jaccard_score",
    "get_accuracy",
    "get_specificity",
    "get_sensitivity",
]


def get_f1_score(pd, gt, threshold=0.5):
    """
    params: pd==prediction
    params: gt: ground truth
    return: dice coefficient / f1-score
    """

    pd = (pd > threshold).float()
    intersection = torch.sum((pd + gt) == 2)
    score = (float(2 * intersection) + 1e-8 )/ (float(torch.sum(pd) + torch.sum(gt)) + 1e-8)

    return score


def get_jaccard_score(pd, gt, threshold=0.5):
    """
    params: pd==prediction
    params: gt: ground truth
    return: jaccard similarity / iou score
    """
    smooth = 1e-8
    pd = (pd > threshold).float()
    intersection = torch.sum((pd + gt) == 2) + smooth
    union = torch.sum((pd + gt) >= 1)

    score = float(intersection ) / (float(union) + smooth)

    return score


def get_accuracy(pd, gt, threshold=0.5):
    """
    params: pd==prediction
    params: gt: ground truth
    return: accuracy score
    """

    pd = (pd > threshold).float()
    corr = torch.sum(pd == gt).float()
    tensor_size = pd.size(0) * pd.size(1) * pd.size(2) * pd.size(3)

    score = float(corr) / float(tensor_size)

    # eps = 1e-8
    # corr = torch.sum(torch.mul(pd, gt)) + torch.sum(torch.mul(1-pd, 1-gt))
    # tensor_size = pd.size(0) * pd.size(1) * pd.size(2) * pd.size(3) + eps
    # score = float(corr) / float(tensor_size)

    return score


def get_sensitivity(pd, gt, threshold=0.5):
    """
    params: pd==prediction
    params: gt: ground truth
    return: sensitivity score / recall rate
    """

    pd = (pd > threshold).float()
    tp = (((pd == 1).float() + (gt == 1).float()) == 2).float()  # True Positive
    fn = (((pd == 0).float() + (gt == 1).float()) == 2).float()  # False Negative

    score = (float(torch.sum(tp)) + 1e-6) / (float(torch.sum(tp + fn)) + 1e-6)
    return score


def get_specificity(pd, gt, threshold=0.5):
    pd = (pd > threshold).float()
    tn = (((pd == 0).float() + (gt == 0).float()) == 2).float()  # True Negative
    fp = (((pd == 1).float() + (gt == 0).float()) == 2).float()  # False Positive

    score = (float(torch.sum(tn))  + 1e-6) / (float(torch.sum(tn + fp)) + 1e-6)
    return score

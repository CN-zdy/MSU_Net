import torch

# SR : Segmentation Result
# GT : Ground Truth
# TP : True Positive
# FN : False Negative
# TN : True Negative
# FP : False Positive

def get_sensitivity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    TP = ((SR==1)+(GT==1))==2
    FN = ((SR==0)+(GT==1))==2

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-8)
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    TN = ((SR==0)+(GT==0))==2
    FP = ((SR==1)+(GT==0))==2

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-8)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    TP = ((SR==1)+(GT==1))==2
    FP = ((SR==1)+(GT==0))==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-8)

    return PC

def get_TPR(SR,GT,threshold=0.5):

    SR = SR > threshold
    GT = GT == torch.max(GT)

    TP = ((SR==1)+(GT==1))==2
    FN = ((SR==0)+(GT==1))==2

    TPR = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-8)

    return TPR

def get_FPR(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    TN = ((SR==0)+(GT==0))==2
    FP = ((SR==1)+(GT==0))==2

    FPR = float(torch.sum(FP))/(float(torch.sum(TN+FP)) + 1e-8)

    return FPR

def get_JS(SR,GT,threshold=0.5):

    SR = SR > threshold
    GT = GT == torch.max(GT)
    
    Inter = torch.sum((SR+GT)==2)
    Union = torch.sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-8)
    
    return JS

def get_DC(SR,GT,threshold=0.5):

    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR + GT) == 2)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-8)

    return DC
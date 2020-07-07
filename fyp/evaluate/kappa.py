import numpy as np
from sklearn.metrics import confusion_matrix

def quadratic_kappa(actuals, preds, N=5, silent = 1):
    """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition
    at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values 
    of adoption rating."""
    try:
        # print(actuals)
        # print(preds)
        w = np.zeros((N,N))
        O = confusion_matrix(actuals, preds)
        if not silent:
            print(O)
            print(np.unique(actuals), [np.mean(np.array(actuals) == j) for j in np.unique(actuals)])        
            print(np.unique(preds), [np.mean(np.array(preds) == j) for j in np.unique(preds)])        

        for i in range(len(w)): 
            for j in range(len(w)):
                w[i][j] = float(((i-j)**2)/(N-1)**2)
        
        act_hist=np.zeros([N])
        for item in actuals: 
            act_hist[item]+=1
        
        pred_hist=np.zeros([N])
        for item in preds: 
            pred_hist[item]+=1
                            
        E = np.outer(act_hist, pred_hist)
        E = E/E.sum()
        O = O/O.sum()
        
        num=0
        den=0
        for i in range(len(w)):
            for j in range(len(w)):
                num+=w[i][j]*O[i][j]
                den+=w[i][j]*E[i][j]
        qk_val = (1 - (num/den))
        return qk_val
    except:
        return -1

a = [0]*20 + [1]*5 + [2]*7 + [3]*3 + [4]
b = [0]*19 + [0] + [1]*4 + [1] + [2]*6 + [2] + [3]*2 + [3] + [2]
# b = np.array([[0,0,1],[0,1,0]])
# print(b)
# print(b.argmax(axis = 1))
x = quadratic_kappa(a,b)
print(x)
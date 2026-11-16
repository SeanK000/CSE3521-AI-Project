import joblib, time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
a=joblib.load(__import__('pathlib').Path(__file__).resolve().parents[1]/'results'/'artifacts.joblib'); X,y=a['Xtr_t'],a['ytr']
cv=StratifiedKFold(3,shuffle=True,random_state=47)
for C in [0.1,0.25,0.5,1]:
    print('svm',C,cross_val_score(LinearSVC(C=C,class_weight='balanced'),X,y,cv=cv,scoring='f1_macro').mean().round(4))
for C in [5,20,50]:
    t=time.time();s=cross_val_score(LogisticRegression(C=C,max_iter=3000,class_weight='balanced'),X,y,cv=cv,scoring='f1_macro',n_jobs=2).mean()
    print('lr',C,round(s,4),round(time.time()-t,1))

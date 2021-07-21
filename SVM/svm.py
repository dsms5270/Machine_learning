# sklearn 라이브러리에서 SVM 분류 모형 가져오기
from sklearn import svm

# 모형 객체 생성 (kernel='rbf' 적용)
# 커널 : 벡터 공간으로 매핑함수
# rbf : radial basis function
# linear
# polynimial
# sigmoid 
svm_model = svm.SVC(kernel='rbf')

#train data를 가지고 모형 학습
svm_model.fit(X_train, y_train)

# test data를 가지고 y_hat을 예측 (분류)
y_hat = svm_model.predict(X_test)
print(y_hat[0:10])
print(y_test.values[0:10])

# 모형 성능 평가 - Confusion 
from sklearn import metrics
svm_matrix = metrics.confusion_matrix(y_test, y_hat)
print(svm_matrix)

# 모형 성능 평가 - 평가지표 계산
svm_report = metrics.classification_report(y_test, y_hat)
print(svm_report)

'''
    confusion_matrix
       
       TP : True Positive 실제값 T, 예측값 T
       FP : False Positive 실제값 F, 예측값T
       FN : False Negative 실제값 T, 예측값 F
       TN : True Negative 실제값 F, 예측값 F

       
  Precision (정확도)
      True로 예측한 대상이 실제 True인 비율 
      정확도가 높다는 것은 FN 오류 작은 경우를 말한다.
  
  Recall (재현율)
      실제값이 True인 분석 대상 중 True로 예측한 비율   
      재현율이 높다는 것은 FN 오류가 낮다는 것을 말한다.
      
   F1-Score (F1지표) 
       정확도와 재현율의 조화 평균을 계산한 값 
       모형의 예측력을 평가 지표 
'''

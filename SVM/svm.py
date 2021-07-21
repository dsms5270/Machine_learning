'''
SVM (서포트 벡터 머신)

- 패턴인식, 자료 분석 등을 위한 지도학습 모델
- 비확룰적 이진 선형 분류 모델을 생성한다.
- SVM의 분류 모델은 데이터들의 경계로 표현되며, 공간상 존재하는 여러 경계 중 가장 큰 폭을 가진 경계를 찾는다.
- 위 그림에서 x와 o의 분류할 때 데이터들 간의 벡터 거리를 측정했을 때, 그 거리가 최대가 되는 분류자를 찾아나간다.
- 각 그룹을 구분하는 분류자를 결정 초평면(Desicion Hyperline), 초평면에 가장 근접한 최전방 데이터들을 서포트벡터라고하며, 서포트 벡터와 초평면 사이의 수직 거리를 마진(Margin)이라고 한다.
- SVM 모형은 선형 분류 뿐만 아니라 비선형 분류에도 사용되는데, 커널 트릭을 사용한다.

- SVM 장점 
1. 분류와 예측에 모두 사용할 수 있다.
2. 신경망 기법에 비해 과적합 정도가 낮다.
3. 예측의 정확도가 높다.
4. 저차원과 고차원의 데이터에 대해서 모두 잘 작동한다.

- SVM 단점 
1. 데이터 전처리와 매개 변수 설정에 따라 정확도가 달라질 수 있다.
2. 예측이 어떻게 이루어지는 지에 대한 이해와 모델에 대한 해석이 어렵다.
3. 대용량 데이터에 대한 모형 구축 시 속도가 느리며, 메모리 할당량이 크다.
'''

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

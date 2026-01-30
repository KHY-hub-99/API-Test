import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 공휴일 데이터 정의 (2025년 기준)
KOREAN_HOLIDAYS_2025 = [
    '20250101', '20250128', '20250129', '20250130', '20250301', 
    '20250303', '20250505', '20250506', '20250606', '20250815', 
    '20251003', '20251005', '20251006', '20251007', '20251008', 
    '20251009', '20251225'
]

# 2. 데이터 로드
population_df = pd.read_csv('./model_data/merged_sdot_data.csv')

# 3. 시간 데이터 변환
population_df['측정시간'] = pd.to_datetime(population_df['측정시간'], format='%Y-%m-%d_%H:%M:%S')
population_df['month'] = population_df['측정시간'].dt.month
population_df['day'] = population_df['측정시간'].dt.day
population_df['hour'] = population_df['측정시간'].dt.hour
population_df['dayofweek'] = population_df['측정시간'].dt.dayofweek

# 4. 공휴일/주말 변수 생성
population_df['date_str'] = population_df['측정시간'].dt.strftime('%Y%m%d')
population_df['is_holiday'] = population_df['date_str'].apply(lambda x: 1 if x in KOREAN_HOLIDAYS_2025 else 0)
population_df['is_weekend'] = population_df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# 5. 타겟 변수(방문자수)를 혼잡도 등급(0, 1, 2)으로 변환
non_zero_data = population_df[population_df['방문자수'] > 0]['방문자수']
q1 = np.percentile(non_zero_data, 33.3)
q2 = np.percentile(non_zero_data, 66.6)

print(f"분류 기준 -> Low: ~{q1:.1f}, Medium: ~{q2:.1f}, High: {q2:.1f} 이상")

def categorize_congestion(count):
    if count <= q1: return 0  # Low
    elif count <= q2: return 1 # Medium
    else: return 2             # High

population_df['congestion_level'] = population_df['방문자수'].apply(categorize_congestion)

# 6. 학습 데이터 준비 (총 8개 피처)
# 자치구 코드가 빠지고 위도/경도가 그 역할을 전적으로 담당합니다.
feature_cols = [
    'month', 'day', 'hour', 'dayofweek',
    'is_holiday', 'is_weekend',
    '위도', '경도'
]
target_col = 'congestion_level'

X = population_df[feature_cols]
y = population_df[target_col]

# 7. 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. 모델 학습
print("\nHistGradientBoosting 모델 학습 중... (속도가 훨씬 빠릅니다)")
model = HistGradientBoostingClassifier(
    learning_rate=0.05,     # 학습률을 낮춰서 더 꼼꼼하게 학습 (기본 0.1)
    max_iter=500,           # 학습 횟수를 늘려서 꼼꼼함을 보완 (기본 100)
    max_leaf_nodes=63,      # 트리의 복잡도를 높여서 미세 패턴 포착 (기본 31)
    min_samples_leaf=100,   # 노이즈에 흔들리지 않도록 최소 샘플 수 증가 (기본 20)
    l2_regularization=1.0,  # 과적합 방지를 위한 규제 추가 (기본 0)
    early_stopping=True,    # 더 이상 성능이 안 오르면 알아서 멈춤
    random_state=42,
    verbose=1               # 학습 과정을 로그로 출력
)
model.fit(X_train, y_train)

# 9. 모델 평가
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n[모델 평가 결과]")
print(f"인구 혼잡도 정확도(Accuracy): {acc:.4f}")
print("-" * 30)
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

# 10. 저장
save_path = './model_data/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# LabelEncoder 저장할 필요 없음! 모델만 저장하면 끝.
joblib.dump(model, f'{save_path}congestion_model_latlon.pkl')

# 기준값 저장 (참고용)
thresholds = {'q1': q1, 'q2': q2}
joblib.dump(thresholds, f'{save_path}congestion_thresholds.pkl')

print(f"모델 저장 완료: {save_path}congestion_model_latlon.pkl")
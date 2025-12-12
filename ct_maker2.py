"""
각 Column 별로 Ct값에 영향을 미치는 과학적 근거
--> ctDNA(circulating tumor DNA)는 종양이 많을수록, 염증이 많을수록, 조직 손상이 많을수록, 전신 대사에 이상이 많을수록 양이 많아진다.
--> 검출하고자 하는 DNA의 양이 많을수록 Ct값은 작아진다.
--> 즉, 몸이 좋지 않는 조건이 많을수록 Ct값이 작아지는 기준으로 점수를 조절하였다.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("Lung Cancer.csv")

mmx_cols = ['MMX1_FAM','MMX1_JA270','MMX2_FAM','MMX2_JA270','MMX3_FAM','MMX3_HEX','MMX3_JA270']
for col in mmx_cols:
    df[col] = None

df['selected'] = np.random.choice(mmx_cols, size=len(df))

# ------------------------
# "cancer_stage" 컬럼의 값을 기준으로 Ct 값을 부여하는 규칙 정의 (랜덤하게)
# Stage I       : 29.0 ~ 32.0
# Stage II      : 27.0 ~ 28.9
# Stage III     : 25.0 ~ 26.9
# Stage IV      : 23.0 ~ 24.9
# ------------------------
def base_ct(stage):
    if stage == 'Stage I':
        return np.random.uniform(29.0, 32.0)
    elif stage == 'Stage II':
        return np.random.uniform(27.0, 28.9)
    elif stage == 'Stage III':
        return np.random.uniform(25.0, 26.9)
    elif stage == 'Stage IV':
        return np.random.uniform(23.0, 24.9)
df['base'] = df['cancer_stage'].apply(base_ct)

# ------------------------
# "smoking_status" 컬럼의 값을 기준으로 Ct 값에 변동을 주는 규칙 정의 (랜덤하게)
# Never Smoked      :  0.0 (기준)
# Passive Smoker    : -0.3
# Former Smoker     : -0.2
# Current Smoker    : -0.8
# ------------------------
smoking_delta = {
    "Never Smoked": 0.0,
    "Passive Smoker": -0.3,
    "Former Smoker": -0.2,
    "Current Smoker": -0.8
}
df['delta_smoking'] = df['smoking_status'].map(smoking_delta)

# ------------------------
# Age : 고령 환자는 tumor mutation burden ↑, ctDNA 양 ↑
#       즉, 고령일수록  Ct ↓
# ------------------------
# <40       : +0.5
# 40 ~ 49   : +0.2
# 50 ~ 59   :  0.0
# 60 ~ 69   : -0.3
# >= 70     : -0.6
# ------------------------
def delta_age(age):
    if age < 40: return +0.5
    elif age < 50: return +0.2
    elif age < 60: return 0.0
    elif age < 70: return -0.3
    else: return -0.6
df['delta_age'] = df['age'].apply(delta_age)

# ------------------------
# Family history : 1차 친족 암 가족력은 유전적 변이/취약성 증가와 관련있다.
#                  종양 진행이 빠른 편이며, ctDNA 증가 가능성이 높다.
# ------------------------
# Yes       : -0.3
# No        :  0.0
# ------------------------
df['delta_family'] = df['family_history'].map({'Yes': -0.3, 'No': 0.0})

# ------------------------
# BMI(유럽 기준으로) : 비만은 전신 염증 증가 + 종양 미세환경 변화 + ctDNA 증가
#                    극저체중 역시 조직 손상의 증가로 인해 ctDNA가 많아질 수 있다는 가능성 고려
# ------------------------
# < 18.5(Underweight)       : -0.2
# 18.5 ~ 24.9(Normal)       :  0.0
# 25.0 ~ 29.9(Overweight)   : -0.2
# >= 30(Obese)              : -0.4
# ------------------------
def delta_bmi(bmi):
    if bmi < 18.5: return -0.2
    elif bmi < 25: return 0.0
    elif bmi < 30: return -0.2
    else: return -0.4
df['delta_bmi'] = df['bmi'].apply(delta_bmi)

# ------------------------
# Cholesterol : 고지혈증은 전신 만성 염증 증가 및 암 관련 pathway가 활성화된다.
# ------------------------
# < 200     :  0.0
# 200 ~ 239 : -0.2
# >= 240    : -0.4
# ------------------------
def delta_chol(ch):
    if ch < 200: return 0.0
    elif ch < 240: return -0.2
    else: return -0.4
df['delta_chol'] = df['cholesterol_level'].apply(delta_chol)

# ------------------------
# Hypertension, Asthma, Cirrhosis(만성질환) : 모두 전신 염증 증가 및 조직 손상과 관련 있음
# ------------------------
# Hypertensia   : -0.2
# Asthma        : -0.2
# Cirrhosis     : -0.5 (간경변은 ctDNA가 가장 증가하는 질환 중 하나로 알려져 있음)
# ------------------------
df['delta_hyper'] = df['hypertension'].map({1: -0.2, 0: 0.0})
df['delta_asthma'] = df['asthma'].map({1: -0.2, 0: 0.0})
df['delta_cirrhosis'] = df['cirrhosis'].map({1: -0.5, 0: 0.0})

# ------------------------
# Noise 추가 (실험 및 검체 상태에 따른 오차 고려)
# ------------------------
noise = np.random.normal(0, 0.7, size=len(df))

# ------------------------
# 최종 Ct 계산
# ------------------------
df['final_ct'] = (
    df['base'].fillna(0)
    + df['delta_smoking'].fillna(0)
    + df['delta_age'].fillna(0)
    + df['delta_family'].fillna(0)
    + df['delta_bmi'].fillna(0)
    + df['delta_chol'].fillna(0)
    + df['delta_hyper'].fillna(0)
    + df['delta_asthma'].fillna(0)
    + df['delta_cirrhosis'].fillna(0)
    + noise
).round(1)

# ------------------------
# 선택된 MMX에 Ct 적용
# ------------------------
for idx, row in df.iterrows():
    df.at[idx, row['selected']] = row['final_ct']

# ------------------------
# 변이 이름 매핑
# ------------------------
mutation_map = {
    'MMX1_FAM': 'Ex19Del',
    'MMX1_JA270': 'S768I',
    'MMX2_FAM': 'L858R',
    'MMX2_JA270': 'T790M',
    'MMX3_FAM': 'L861Q',
    'MMX3_HEX': 'G719X',
    'MMX3_JA270': 'Ex20Ins'
}
df['mutation_detected'] = df['selected'].map(mutation_map)

from datetime import datetime
# 파일 이름 뒤에 실행한 날짜+시간 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
output_filename = f"Lung Cancer_CT_{timestamp}.csv"
df.to_csv(output_filename, index=False)
print("저장 완료:", output_filename)

# 성별 별 검출 변이 분포
df.groupby(['gender', 'mutation_detected']).size().unstack().plot(kind='bar')

# 연령 별 검출 변이 분포 (10세 단위로)
df['age_bin'] = (df['age'] // 10) *10
df.groupby(['age_bin', 'mutation_detected']).size().unstack().plot(kind='bar')


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 변이 분포 별 bar plot
mutation_counts = df['mutation_detected'].value_counts().sort_index()

plt.figure(figsize=(8, 5))
mutation_counts.plot(kind='bar')
plt.title('검출 변이 빈도')
plt.xlabel('EGFR Mutation')
plt.ylabel('환자 수')
plt.tight_layout()
plt.show()

# 변이 별 Ct값 분포 (Box plot)
plt.figure(figsize=(10, 6))
ax = df.boxplot(column='final_ct', by='mutation_detected')
positions = range(1, len(df['mutation_detected'].unique())+1)
medians = df.groupby('mutation_detected')['final_ct'].median().values
for pos, median in zip(positions, medians):
    ax.text(
        pos, median + 0.3,
        f"{median:.1f}",
        horizontalalignment='center',
        color='black',
        fontsize=10,
        fontweight='bold'
    )
plt.title("EGFR Mutation 별 Ct값 분포")
plt.suptitle("")
plt.xlabel("EGFR Mutation")
plt.ylabel("Ct value")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# cancer_stage별 mutation별 Ct값 Boxplot
import seaborn as sns

stage_order = ["Stage I", "Stage II", "Stage III", "Stage IV"]

df["cancer_stage"] = pd.Categorical(
    df["cancer_stage"],
    categories=stage_order,
    ordered=True
)

plt.figure(figsize=(12, 7))
sns.boxplot(
    data=df,
    x="mutation_detected",
    y="final_ct",
    hue="cancer_stage"
)

plt.title("EGFR Mutation X Cancer stage 별 Ct값 분포")
plt.xlabel("EGFR Mutation")
plt.ylabel("Ct value")
plt.legend(title="Cancer stage")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
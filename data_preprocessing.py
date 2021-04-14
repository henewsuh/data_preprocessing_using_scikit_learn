
#%% 1. 라이브러리 및 데이터 불러오기 

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv('high_diamond_ranked_10min.csv')
df_head = df.head() 


#%% 2. 범주형/수치형 데이터 분리

df.drop(['gameId', 'redFirstBlood', 'redKills', 'redDeaths',
       'redTotalGold', 'redTotalExperience', 'redGoldDiff',
       'redExperienceDiff'], axis=1, inplace=True)


X_num = df[['blueWardsPlaced', 'blueWardsDestroyed', 
       'blueKills', 'blueDeaths', 'blueAssists', 'blueEliteMonsters',
       'blueTowersDestroyed', 'blueTotalGold',
       'blueAvgLevel', 'blueTotalExperience', 'blueTotalMinionsKilled',
       'blueTotalJungleMinionsKilled', 'blueGoldDiff', 'blueExperienceDiff',
       'blueCSPerMin', 'blueGoldPerMin', 'redWardsPlaced', 'redWardsDestroyed',
       'redAssists', 'redEliteMonsters', 'redTowersDestroyed', 'redAvgLevel', 'redTotalMinionsKilled',
       'redTotalJungleMinionsKilled', 'redCSPerMin', 'redGoldPerMin']]

X_cat = df[['blueFirstBlood', 'blueDragons', 'blueHeralds', 'redDragons', 'redHeralds']]
y = df['blueWins']


#%% 3. 수치형 데이터 스케일링 

scaler = StandardScaler()
scaler.fit(X_num)
X_scaled = scaler.transform(X_num)
X_scaled = pd.DataFrame(X_scaled, index=X_num.index, columns=X_num.columns)

X = pd.concat([X_scaled, X_cat], axis=1)


#%% 4. 범주형 데이터 트랜스폼 

X_catt = pd.get_dummies(X_cat, columns=['blueFirstBlood'], drop_first=True)



#%% 5. 학습데이터/테스트 데이터 분리하기 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



#%% 6. 분류 모델 학습

model_lr = LogisticRegression(max_iter=10000)
model_lr.fit(X_train, y_train)


#%% 7. 모델 학습 결과 평가하기 

pred = model_lr.predict(X_test)
print(classification_report(y_test, pred))













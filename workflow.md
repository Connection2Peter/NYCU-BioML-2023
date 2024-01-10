# TODO

## Experiment
### 1. 建模方法 & Windows & DB Ratio & Normalization
#### 固定參數
1. Feature set 
    - SSE-PSSM
2. Database
    - 全序列計算後拆分
#### 狀態
- OK

#### 結果
- RF XGB CAT 霸榜
- 要 Normalization
- 以 AUC 評斷

### 2. Feature combination
#### 固定參數
1. Model
    - RF
2. Windows size
    - 21
    - 55
3. Ratio
    - 1

#### 狀態
- 整合建立 Feature 的 Code
    - 主程式 (Connection)
        - feature selection
            1. 單獨評斷
                - 30 -> 5
            2. 組和計算
            3. 列出前 5 組合，進入 Experiment 3
    - Feature
        - Brian
            1. PWM (P, N, P+N, P-N)
            2. 正負電
            3. 極性
        - Ivern
            1. EAAC
            2. CKSAAP
            3. DPC (又是你，你最爛)
            4. DDE
        - Other
            1. SSE-PSSM
    - 專案位置
        - ```~/local/connection/NYCU-2023-BioML```
    - 修改方式
        - ```cp {檔案原名} {修改人}_{檔案原名}```
    - 可用機器
        - 240 Mothra (24 core)
- 完成 ID Test 的 Code
    1. 讀檔 : *.tsv
    2. 計算 Feature
        - 完成分算運算 (等待確定)
    3. Prediction

#### 結果
- 主程式 (Connection) 尚未完成

### 3. Hyper parameter tuning
#### 最終調教
1. Model (s)
    - RandomForest
    - CatBoost
    - XGBoost
2. Windows size
    - 40
    - 80
3. Ratio
    - 1
4. Feature
    - ???
5. Hyper parameter tuning
    - ???

#### 狀態
- 一台測一個 Model
    - RandomForest : Mothra
    - CatBoost     : VR
    - XGBoost      : Sphinx
- 完成 code

#### 結果
- 待完成程式

## Hypothesis
- Method
    1. Seq -> k-mer -> feature
    2. Seq -> feature -> k-mer
- 理論
    - 原因
        - 發現 Positive 以及 Negative 內都具有相同序列
    - 問題
        - Positive 以及 Negative 內相同序列理論上會產生相同 Feature
    - 假設
        - 如果先計算 feature，再基於不同位置切 K-mer，Positive 以及 Negative 內相同序列理論上會產生不同 Feature
        - 不同的 Feature 就可以分別出 Positive or Negative
    - 驗證實驗
        - 在 DB 中紀錄所有相同的序列位置，從 Methods 2 中撈出對應的 Feature matrix
        - 已驗證

## DB-DB Similarity
- 驗證其中一個優化方式
    - cd-hit-2d NR deduplication
        - 確實有用，不過需要圖表佐證
        - 大約平均 AUC +3%
    - 驗證實驗
        1. 把Positive & Negative 倆倆之間做相似度比對
        2. 以 40 ~ 100 identity 挑出對應的 Seqs
        3. 計算對應向量
        4. 做成 Graph

## PPT
1. Data preparation and preprocessing
2. Features investigation
3. Model selection and performance evaluation
4. Perspectives (bonus grading)

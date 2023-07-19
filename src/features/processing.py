import os
os.chdir('/Users/sizhechen/Desktop/git/fatty-liver-project/src/features')

import sys
sys.path.append('/Users/sizhechen/Desktop/git/fatty-liver-project/src/utilities')
print(sys.path)

from utilities import utils
print("start reading csv file")

#Imports
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# read dataset as df
df = pd.read_excel("/Users/sizhechen/Desktop/git/fatty-liver-project/src/data/NTCMRC_all.xlsx")

# Create a copy of df as df1
df1 = df.copy()

# Replace '\\N' with NaN
df1 = df1.replace('\\N', np.nan)

# Specify the columns to be converted to FLOAT
columns_to_convert1 = ['BMI', 'Triglyceride_y', 'gamgt', 'waist_y', 'mst', 'egfrn', 'Estimated_GFR_x', 'Alb_Cre_ratio', 'HOMA_IR', 'HS_CRP', \
                       'LDL_C_direct', 'LDL_C_HDL_C', 'Adiponectin', 'Leptin', 'Uric_Acid','Insulin', 'ALT_GPT']

# Specify the columns to be converted to INT
columns_to_convert2 = ['脂肪肝 fatty Liver (0:正常  1:mild 2:moderate 3:severe)', 'smoke', 'smoke_q', \
                       'sex', 'w', 'coffee', 'betel']

# Convert the specified columns to float and fill missing/unconvertible values with NaN
for column in columns_to_convert1:
    df1[column] = pd.to_numeric(df1[column], errors='coerce')

# Convert the specified columns to int and fill missing/unconvertible values with NaN
for column in columns_to_convert2:
    df1[column] = pd.to_numeric(df1[column], errors='coerce').astype(pd.Int64Dtype())

# Calculate FLI using the formula and defined as df2
df2 = df1.copy()
df2['FLI'] = (np.exp(0.953 * np.log(df2['Triglyceride_y']) + 0.139 * df2['BMI'] + 0.718 * np.log(df2['gamgt']) \
     + 0.053 * df2['waist_y'] - 15.745)) / (1 + np.exp(0.953 * np.log(df2['Triglyceride_y']) \
    + 0.139 * df2['BMI'] + 0.718 * np.log(df2['gamgt']) + 0.053 * df2['waist_y'] - 15.745)) * 100

# Derive FL_echo based on ultrasound results column
df2['FL_echo'] = df2['脂肪肝 fatty Liver (0:正常  1:mild 2:moderate 3:severe)']
df2['FL_echo'] = df2['FL_echo'].replace('<NA>', np.nan)
df2['fl_status'] = df2.apply(utils.derive_fl_status, axis=1)

#Derive homa_ir_check, hs_crp_check, and mst_total to determine MAFLD risk factors
df2['homa_ir_check'] = df2['HOMA_IR'].apply(lambda x: 1 if x >= 2.5 else 0)
df2['hs_crp_check'] = df2['HS_CRP'].apply(lambda x: 1 if x > 2 else 0)
df2['mst_total'] = df2[['w', 'hyper', 'HDL', 'fg', 'trig', 'homa_ir_check', 'hs_crp_check']].sum(axis=1)


df3 = utils.derive_CKD(utils.derive_MAFLD(df2))

# Derive FL_group_list, 对CMRC_id分组并计算每个病人的FL_Check的唯一值
grouped = df2.groupby('CMRC_id')['fl_status'].unique()
df2['fl_group_list'] = df2.groupby('CMRC_id')['fl_status'].transform(lambda x: [x.unique().tolist()] * len(x))

#assign patient valid value
df4 = utils.assign_patient_fl_validity(df3)

# Deal with values with both string and num values: HBsAg_x, Anti_HCV_x, (values eg. 陰性    0.351)
# extract numeric values for these cols and rename that as  *_num
columns_to_extract = ['HBsAg_x', 'Anti_HCV_x']
df5 = df4.copy()
for column in columns_to_extract:
    new_column_name = column + '_num'
    df5[new_column_name] = df5[column].apply(utils.extract_numeric_value)

# Sliding window implementation, ie. use previous 2 years record to predict the 3 year MAFLD status
df6 = utils.sliding_window_data(df5, input_window_size=2, target_window_size=1)

# Filter available data that can be applied to models
df7 = df6[(df6['t3_MAFLD'] != -1)]

# Drop ID relevant cols in the dataset
columns_to_drop = ['CMRC_id', 't1_CMRC_id', 't2_CMRC_id', 't1_sid', 't2_sid', 't1_P_Number','t2_P_Number']
df8 = df7.drop(columns=columns_to_drop)

#Select key columns for conventional machine learning models
columns = ["sex", "age", "waist_y", "Glucose_AC_y", "Triglyceride_y", "HDL_C_y", "AST_GOT", "ALT_GPT", \
          "gamgt", "Insulin", "T_Cholesterol", "LDL_C_direct", "VLDL_C", "Non_HDL_C", "T_CHOL_HDL_C", \
          "LDL_C_HDL_C", "HS_CRP", "Hb_A1c", "Uric_Acid", "HBsAg_x", "Anti_HCV_x", "HOMA_IR", "Adiponectin", \
           "Leptin", "TotalVitaminD", "smoke", "smoke_q", "coffee", "betel", "BMI", "DM_determine", "w", "hyper", \
           "fg", "HDL", "trig", "sarcf", "ms2", "MNA", "AUDIT", "HBV_", "HCV_", "MAFLD", "CKD", \
           'HBsAg_x_num', 'Anti_HCV_x_num']
prefixes = ["t1_", "t2_"]
renamed_columns = utils.add_prefix(columns, prefixes)

df9 = df8[renamed_columns]
df9['t3_MAFLD'] = df8['t3_MAFLD']

# drop these cols as those been derived for numeric cols, remain alias *_num,
cols_to_drop_only_MAFLD = ['t1_HBsAg_x', 't2_HBsAg_x', 't1_Anti_HCV_x', 't2_Anti_HCV_x', 't1_MAFLD', 't2_MAFLD']
cols_to_drop_fli_related = ['t1_HBsAg_x', 't2_HBsAg_x', 't1_Anti_HCV_x', 't2_Anti_HCV_x', 't1_MAFLD', 't2_MAFLD', \
                            't1_Triglyceride_y', 't1_BMI', 't1_gamgt', 't1_waist_y', 't1_gamgt', 't1_w', \
                            't2_Triglyceride_y', 't2_BMI', 't2_gamgt', 't2_waist_y', 't2_gamgt', 't2_w']
df9_a = df9.drop(cols_to_drop_only_MAFLD, axis=1)

#FLI related cols: Triglyceride_y, BMI, gamgt, waist_y, gamgt
df9_b = df9.drop(cols_to_drop_fli_related, axis=1)

print(df9_a.head(1))

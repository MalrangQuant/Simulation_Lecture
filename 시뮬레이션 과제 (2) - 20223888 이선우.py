import numpy as np
import pandas as pd
from IPython.display import display

'''---------------= INPUT DATA ----------------'''

num = 10000
num_asset = 2
step = 32
T = 2
dt = T / step #/num_time_step
r = 0.03 
q = 0
rho = 0.2
sigma1 = 0.3 
sigma2 = 0.4
rate = 0.125

s1_0 = 100
s2_0 = 100

case_dict = {
    1 : 100 * (1 + 0.5 * rate) / (1 + 0.5 * r),
    2 : 100 * (1 + 1.0 * rate) / (1 + 1.0 * r),
    3 : 100 * (1 + 1.5 * rate) / (1 + 1.5 * r),
    4 : 100 * (1 + 2.0 * rate) / (1 + 2.0 * r),
    5 : 100 / (1 + 2.0 * r),
    6 : 1 / (1 + 2.0 * r)
}

'''---------------- Equations and Simulation -----------------'''

path1 = np.ones((num, 1)) * s1_0
path2 = np.ones((num, 1)) * s2_0

z1 = np.random.normal(size=(step, num)).T
z2 = rho * z1 + np.sqrt(1-rho**2) * np.random.normal(size=(step, num)).T

x1 = np.exp( (r - q - 0.5 * (sigma1 **2)) * dt + sigma1 * np.sqrt(dt) * z1)
x2 = np.exp( (r - q - 0.5 * (sigma1 **2)) * dt + sigma1 * np.sqrt(dt) * z2)

for i in range(step):
    next_path = x1[:,i].reshape(num, 1) * path1[:, -1].reshape(num, 1)
    path1 = np.c_[path1, next_path]

for i in range(step):
    next_path = x2[:,i].reshape(num, 1) * path2[:, -1].reshape(num, 1)
    path2 = np.c_[path2, next_path]

path1_df = pd.DataFrame(data=path1, index=[x for x in range(num)], columns=[x for x in range(step+1)])
path2_df = pd.DataFrame(data=path2, index=[x for x in range(num)], columns=[x for x in range(step+1)])

check1 = pd.DataFrame(data=path1, index=[[x for x in range(num)],[0 for x in range(num)]], columns=[x for x in range(step+1)])
check2 = pd.DataFrame(data=path2, index=[[x for x in range(num)],[1 for x in range(num)]], columns=[x for x in range(step+1)])


df = pd.concat([check1, check2]).sort_index()
display(df)


'''----------------------- Case by Case --------------------------'''

'''Case1 Check'''
'''모두 6개월 시점에서 85 이상일때'''
case1_arr = (((path1_df[8] >= 85) * 1) * ((path2_df[8] >= 85) * 1))

'''Case2 Check'''
'''모두 12개월 시점에서 80 이상일때'''
case2_arr = (((path1_df[16] >= 80) * 1) * ((path2_df[16] >= 80) * 1))
case2_arr = np.where((case2_arr - case1_arr) > 0, 1, 0)

'''Case3 Check'''
'''모두 16개월 시점에서 75 이상일때'''
case3_arr = (((path1_df[24] >= 75) * 1) * ((path2_df[24] >= 75) * 1))
case3_arr = np.where((case3_arr - case2_arr - case1_arr) > 0, 1, 0)

'''Case1, Case2, Case3 조기상환'''
early_redemption_arr = (((((case1_arr - 1) ** 2) * ((case2_arr - 1) ** 2) * ((case3_arr - 1) ** 2)) - 1) ** 2)

'''Case4 Check'''
'''모두 24개월 시점에서 70 이상일때'''
case4_arr = (((path1_df[32] >= 70) * 1) * ((path2_df[32] >= 70) * 1))
case4_arr = np.where(case4_arr - case3_arr - case2_arr - case1_arr > 0, 1 ,0)

'''Case5 Check, 조기상환 경우는 제외함'''
'''둘다 한번도 60 미만으로 하락한 적이 없고, 만기 평가 가격이 한 종목 이라도 70 미만인 경우'''
case5_arr = ( (((path1_df >= 60).prod(axis=1) * (path2_df >= 60).prod(axis=1))) * (((((path1_df[32] >= 70) * 1) * ((path2_df[32] >= 70) * 1)) - 1) ** 2) )
case5_arr = np.where((case5_arr - early_redemption_arr) > 0, 1, 0)

'''Case6 Check, 조기상환 경우는 제외함'''
'''둘다 한번이라도 60 미만으로 하락한 적이 있고, 만기 평가 가격이 한 종목 이라도 70 미만인 경우'''
case6_arr = ((((path1_df >= 60).prod(axis=1) * (path2_df >= 60).prod(axis=1)) - 1 ) **2) * (((((path1_df[32] >= 70) * 1) * ((path2_df[32] >= 70))) - 1) **2)
case6_arr = np.where((case6_arr - early_redemption_arr) > 0, 1, 0)

case = case1_arr * 1 + case2_arr * 2 + case3_arr * 3 + case4_arr * 4 + case5_arr * 5 + case6_arr * 6

'''--------------------- Values from case by case -------------------------'''

path1_df['case'] = case
path2_df['case'] = case


path1_df['Y'] = np.where(path1_df['case'] == 1, case_dict[1],\
                np.where(path1_df['case'] == 2, case_dict[2],\
                np.where(path1_df['case'] == 3, case_dict[3],\
                np.where(path1_df['case'] == 4, case_dict[4],\
                np.where(path1_df['case'] == 5, case_dict[5],\
                np.where(path1_df['case'] == 6, case_dict[6] * min( s1_0 * (path1_df[step][3] / 100), s2_0 * (path2_df[step][3] / 100)), 0))))))

path2_df['Y'] = path1_df['Y']

display(path1_df[['case', 'Y']])


'''------------------------- Price and Standard, and DF sorting -------------------------'''

value = path2_df['Y'].mean()
sample_std = np.sqrt(sum((path2_df['Y'] - value)**2)/(num-1))
mc_simul_error = sample_std/ np.sqrt(num)

prob_df = pd.DataFrame(data=path1_df.groupby('case').size().index, columns=['case']).set_index('case')
prob_df['count'] = path1_df.groupby('case').size()
prob_df['prob(%)'] = (path1_df.groupby('case').size() / num) * 100

display(prob_df)

print(f'ELS price = {value}, standard error = {mc_simul_error}')

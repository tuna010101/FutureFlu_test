# -*- coding: utf-8 -*-
"""Linear mutation-prevalence prediction for FutureFlu subclade seasons.

English: Fit historical HA1 site prevalences and score candidate mutations for
a target season / hemisphere / subtype.
中文：基于历史 HA1 位点流行度拟合，并对目标季/半球/亚型的候选突变打分。
"""

import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations
import re
import os


def site_prevalence(seq, predict_season, semisphere, subtype):
    """
    计算氨基酸位点流行度
    """
    # 定义HA1区域范围
    ha1_ranges = {
        'H3N2': (17, 345),
        'H1N1': (18, 344),
        'Victoria': (16, 362)
    }
    ha1_range = ha1_ranges[subtype]
    
    years = sorted([y for y in seq['season'].unique() if y > 2009 and y < predict_season])
    
    AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
    
    # 数据预处理
    seq = seq.copy()
    seq['submission_date'] = seq['submission_date'].fillna(seq['collection_date'])
    seq['collection_date'] = pd.to_datetime(seq['collection_date'])
    seq['submission_date'] = pd.to_datetime(seq['submission_date'])
    
    # 筛选HA1区域的位点
    site_columns = seq.filter(regex='^X\d+').columns
    site_columns = [col for col in site_columns 
                   if ha1_range[0] <= int(re.findall(r'\d+', col)[0]) <= ha1_range[1]]
    
    for col in site_columns:
        seq[col] = seq[col].str.upper()

    annual_data = {}
    prevalence_data = []
    
    def get_consensus_sequence(data, site_columns):
        """计算一致性序列"""
        consensus = {}
        for col in site_columns:
            valid_data = data[col][data[col] != 'X']
            if len(valid_data) > 0:
                consensus[col] = valid_data.mode().iloc[0]
        return consensus
    
    def calculate_significance(new_aa_current, new_aa_prev, total_current, total_prev):
        """计算卡方检验的p值 - 检验new_aa在两个季节中的分布差异"""
        from scipy import stats
        import numpy as np
        
        # 构建2x2列联表：[new_aa, other_aa] x [current_season, prev_season]
        observed = np.array([
            [new_aa_current, total_current - new_aa_current],  # current season
            [new_aa_prev, total_prev - new_aa_prev]            # previous season
        ])
        
        # 避免零值导致的计算错误
        if np.any(observed == 0):
            return 0.0
            
        _, p_value = stats.chi2_contingency(observed)[:2]
        return p_value

    for year in years:
        # 确定时间范围
        if semisphere == "North":
            start = f"{year}-09-01"
            end = f"{year+1}-02-01"
            submission_end = f"{predict_season}-02-01"

        else:
            start = f"{year}-02-01"
            end = f"{year}-09-01"
            submission_end = f"{predict_season-1}-09-01"
            
        # 筛选季节数据
        season_data = seq[
            (seq['collection_date'] >= start) &
            (seq['collection_date'] < end) &
            (seq['submission_date'] < submission_end)
        ]
        
        # === 计算氨基酸频率 ===
        year_freq_data = {}
        year_counts_data = {}
        for site_col in site_columns:
            valid_data = season_data[site_col][season_data[site_col] != 'X']
            freq = valid_data.value_counts(normalize=True)
            counts = valid_data.value_counts()
            for aa in AMINO_ACIDS:
                col_name = f"{site_col}{aa}"
                year_freq_data[col_name] = freq.get(aa, 0.0)
            year_counts_data[site_col] = counts
        
        prevalence_data.append(pd.Series(year_freq_data, name=year))
        
        # === 初始化年度数据 ===
        annual_data[year] = {
            'freqs': year_freq_data,
            'counts': year_counts_data,
            'dominant_mutations': []
        }
        
        # === 识别优势突变 ===
        if year != years[0]:  # 跳过第一年
            # 获取上一年的数据
            if semisphere == "North":
                prev_start = f"{year-1}-09-01"
                prev_end = f"{year}-02-01"
            else:
                prev_start = f"{year-1}-02-01"
                prev_end = f"{year-1}-09-01"
            
            prev_data = seq[
                (seq['collection_date'] >= prev_start) &
                (seq['collection_date'] < prev_end) &
                (seq['submission_date'] < submission_end)
            ]
            
            # 计算一致性序列
            current_consensus = get_consensus_sequence(season_data, site_columns)
            prev_consensus = get_consensus_sequence(prev_data, site_columns)
            
            # 获取前一年的氨基酸计数
            prev_counts_data = {}
            for site_col in site_columns:
                valid_data = prev_data[site_col][prev_data[site_col] != 'X']
                prev_counts_data[site_col] = valid_data.value_counts()
            
            identified_mutations = []
            for site_col in site_columns:
                if (site_col in current_consensus and site_col in prev_consensus and
                    current_consensus[site_col] != prev_consensus[site_col]):
                    
                    old_aa = prev_consensus[site_col]
                    new_aa = current_consensus[site_col]
                    
                    # 获取计数数据
                    current_counts = year_counts_data[site_col]
                    prev_counts = prev_counts_data[site_col]
                    
                    new_aa_current = current_counts.get(new_aa, 0)
                    new_aa_prev = prev_counts.get(new_aa, 0)
                    total_current = sum(current_counts)
                    total_prev = sum(prev_counts)
                    
                    # 进行卡方检验
                    p_value = calculate_significance(new_aa_current, new_aa_prev, 
                                                   total_current, total_prev)
                    # if subtype == 'H3N2' or subtype == 'Victoria':
                    if p_value < 0.05:
                        mut = f"{site_col[1:]}{new_aa}"
                        identified_mutations.append(mut)

            annual_data[year]['dominant_mutations'] = sorted(identified_mutations)

    # === 结果格式化 ===
    prevalence_result = pd.concat(prevalence_data, axis=1).T
    
    # 添加结果列
    prevalence_result['dominant_mutation'] = [
        ', '.join(annual_data[y]['dominant_mutations']) for y in years
    ]
       
    # 调整列顺序
    mutation_columns = [col for col in prevalence_result.columns 
                       if col not in ['dominant_mutation', 'dominant_clades']]
    column_order = mutation_columns + ['dominant_mutation']
    
    prevalence_result = prevalence_result[column_order]
    
    return prevalence_result.reset_index().rename(columns={'index': 'season'})
    
########################################
# 2. 计算总流行度
########################################
def gmeasure(prev_data, theta_range):
    """计算每个theta参数下的年度gsum总和
    
    参数:
        prev_data (DataFrame): site_prevalence函数的输出，包含各年份位点流行度
        theta_range (list): theta参数列表（如[0.1, 0.2]）
    
    返回:
        DataFrame: 结构为
            - 行: 各年份
            - 列: season | theta=0.10 | theta=0.15 | ...
    """
    # ======================================================
    # 第一部分：初始化数据结构
    # ======================================================
    # 直接获取所有年份（假设数据中无transition_time行）
    valid_years = prev_data['season'].tolist()
    year_theta_gsum = {year: {theta: 0.0 for theta in theta_range} for year in valid_years}
    
    # ======================================================
    # 第二部分：遍历每个theta和位点，计算gsum
    # ======================================================
    for theta in theta_range:
        for col in prev_data.columns:
            # 仅处理位点列（格式：X156K）
            if col.startswith('X'):
            # 获取流行度序列并处理缺失值
                values = prev_data[col].fillna(0).values
                n_years = len(values)
                
                # 初始化突变标记
                mut = np.zeros(n_years, dtype=int)
                start = 0  # 检测窗口起始
                
                # 滑动窗口检测突变
                for r in range(n_years):
                    # 条件1：当前值≥theta
                    # 条件2：窗口内存在低值（<0.1）
                    if values[r] >= theta and np.any(values[start:r] < theta):
                        low_pos = np.where(values[:r] < theta)[0]
                        if low_pos.size > 0:
                            a = low_pos[-1]
                            # 标记突变区间（a+1到r）
                            mut[a+1:r+1] = 1
                            # 移动窗口到当前突变结束之后
                            start = r + 1
                
                # 计算年度gsum并累加
                yearly_gsum = values * mut
                for idx, year in enumerate(valid_years):
                    year_theta_gsum[year][theta] += yearly_gsum[idx]
    
    # ======================================================
    # 第三部分：构建结果DataFrame
    # ======================================================
    # 转换年度gsum数据
    gsum_df = pd.DataFrame.from_dict(year_theta_gsum, orient='index')
    gsum_df.columns = [f'theta={theta:.2f}' for theta in theta_range]
    gsum_df = gsum_df.reset_index().rename(columns={'index': 'season'})
    
    return gsum_df


def setup_logging(base_path):
    """设置日志记录
    Args:
        base_path: 基础路径
    Returns:
        日志写入函数
    """
    # os.path.join 用于拼接路径
    log_file = os.path.join(base_path, "best_result_para.txt")
    
    # 定义一个只写入文件的函数
    def log_print(*args, **kwargs):
        # log_print 函数使用 print 但将输出重定向到文件，初始化日志文件时写入开始时间和分隔线等。
        with open(log_file, 'a', encoding='utf-8') as f:
            # *args：接受任意数量的位置参数（如多个字符串）,**kwargs：接受任意关键字参数（如 sep=' ', end='\n'）
            print(*args, **kwargs, file=f)
    
    # 初始化日志文件
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"分析开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n")
    
    return log_print


########################################
# 3. Fitness拟合
########################################

def fit_regression(input_gmeasure, input_epi_data, log_print):
    """拟合回归以找到最优参数
    Args:
        input_gmeasure (DataFrame): gmeasure函数的输出，包含各年份和各theta的gsum值
        input_epi_data (DataFrame): 流行病学数据，需包含Season和Positivity_Rate列
        log_print (function): 日志记录函数
    
    Returns:
        tuple: (最佳theta值, 最佳R²值, 使用的年份数)
    """
    # 导入必要的库
    import statsmodels.api as sm
    from statsmodels.genmod.families import Binomial

    # === 数据预处理 ===
    gmeasure = input_gmeasure.set_index('season').copy()
    epi_data = input_epi_data.copy()
    
    # 统一数据类型
    epi_data['Season'] = epi_data['Season'].astype(int)
    gmeasure.index = gmeasure.index.astype(str)
    
    # === 数据对齐 ===
    valid_gmeasure_years = [y for y in gmeasure.index if y.isdigit()]
    gmeasure_years = pd.to_numeric(valid_gmeasure_years, errors='coerce')
    gmeasure_years = gmeasure_years[~np.isnan(gmeasure_years)].astype(int).tolist()
    
    # 获取共同时序范围
    common_years = sorted(set(epi_data['Season']).intersection(gmeasure_years))
    
    # 记录最优结果
    best_overall = {
        'theta': None,
        'r_squared': -float('inf'),
        'years_used': None,
        'start_year': None
    }
    
    # === 不同年份范围的拟合 ===
    max_year = max(common_years)
    log_print("\n开始不同年份范围的拟合分析:")
    
    # 从最少3年开始，逐步增加年份数直到包含所有历史年份
    for years_back in range(3, len(common_years) + 1):
        start_year = max_year - years_back + 1
        current_years = [y for y in common_years if start_year <= y <= max_year]
        
        if len(current_years) < 3:  # 确保至少有3年数据
            continue
            
        log_print(f"\n使用{years_back}年数据进行拟合 ({start_year}-{max_year}):")
        
        # 筛选对应年份的数据
        current_epi = epi_data[epi_data['Season'].isin(current_years)]
        current_seq = gmeasure.loc[[str(y) for y in current_years]]
        
        # === 参数扫描 ===
        results = []
        for col in current_seq.columns:
            if not col.startswith('theta='):
                continue
            
            theta = float(col.split('=')[1])
            
            # 准备回归数据
            X = current_seq[col].values.reshape(-1, 1)
            
            # 获取阳性率并转换为0-1之间的比例值
            y_raw = current_epi['Positivity_Rate'].values
            y = y_raw / 100.0  # 将百分数转换为比例值（如50转为0.5）
            
            # 确保阳性率在0-1范围内，避免极端值导致计算问题
            y = np.clip(y, 0.0001, 0.9999)
            
            # 添加截距项
            X = sm.add_constant(X)
            
            try:
                # 使用带logit链接函数的广义线性模型
                model = sm.GLM(y, X, family=Binomial(link=sm.families.links.logit()))
                result = model.fit(disp=0)  # 设置disp=0禁止打印优化信息
                
                # 获取伪R²
                null_model = sm.GLM(y, np.ones(len(y)).reshape(-1, 1), family=Binomial(link=sm.families.links.logit()))
                null_result = null_model.fit(disp=0)
                r_squared = 1 - (result.deviance / null_result.deviance)
                
                results.append({
                    'Theta': theta,
                    'R_squared': round(r_squared, 3)
                })
                
            except Exception as e:
                log_print(f"拟合theta={theta}时出错: {str(e)}")
                continue
        
        # === 分析当前年份范围的结果 ===
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(['R_squared', 'Theta'], ascending=[False, True])
            best_current = results_df.iloc[0]
            
            log_print(f"当前最佳 - Theta: {best_current['Theta']:.2f}, R²: {best_current['R_squared']:.3f}")
            
            # 更新全局最优结果
            if best_current['R_squared'] > best_overall['r_squared']:
                best_overall['theta'] = best_current['Theta']
                best_overall['r_squared'] = best_current['R_squared']
                best_overall['years_used'] = years_back
                best_overall['start_year'] = start_year
    
    # === 输出最终结果 ===
    if best_overall['theta'] is not None:
        log_print("\n=== 最终最优拟合结果 ===")
        log_print(f"使用年份范围: {best_overall['start_year']}-{max_year} ({best_overall['years_used']}年)")
        log_print(f"最优Theta: {best_overall['theta']:.2f}")
        log_print(f"最优R²: {best_overall['r_squared']:.3f}")
        return best_overall['theta'], best_overall['r_squared'], best_overall['years_used']
    else:
        log_print("\n警告: 未找到有效的拟合结果")
        return None, None, None
########################################
# 4. 预测下一年的风险突变
########################################
    
def predict_mutations_multi_model(predict_season, theta, prev_data, mutation_history, output_file=None):
    """
    使用四种不同模型预测指定年份的风险突变，并将所有结果保存到一个文件
    
    参数:
        predict_season (int): 需要预测的流行季（如2023）
        theta (float): 风险阈值，来自fit_regression函数的最佳结果
        prev_data (DataFrame): site_prevalence函数的输出，包含历史流行度
        mutation_history (DataFrame): risk_mutations函数的输出，包含历史突变记录
        output_file (str, optional): 结果输出文件路径，如果为None则不保存文件
        
    返回:
        DataFrame: 包含四种模型预测的综合结果，有一个model列标识使用的预测模型
    """
    import os
    import pandas as pd
    import numpy as np
    
    # 定义要使用的模型
    models = ['linear']
    
    # 存储所有预测结果的DataFrame
    # 修改：增加了delta列在columns中
    all_predictions = pd.DataFrame(columns=['predict_season', 'risk_mutation', 'previous_prevalence', 'predicted_prevalence', 'delta', 'model'])
    
    # === 数据准备 ===
    # 获取预测年份前所有历史数据
    mutation_columns = [col for col in prev_data.columns if col.startswith('X')]
    historical_data = prev_data[prev_data.index < predict_season][mutation_columns]    

    # === 构建突变历史字典 ===
    # 记录每个突变成为dominant mutation的最晚年份
    mutation_dominant_years = {}
    if 'dominant_mutation' in prev_data.columns:
        for idx, row in prev_data.iterrows():
            year = row.name if hasattr(row, 'name') else prev_data.iloc[idx]['season']
            if year >= predict_season:
                continue
                
            dominant_muts = row['dominant_mutation']
            if isinstance(dominant_muts, str) and dominant_muts.strip():
                for mut in dominant_muts.split(','):
                    mut = mut.strip()
                    if mut:
                        # 如果这个突变还没有记录，或者当前年份更晚，则更新记录
                        if mut not in mutation_dominant_years or year > mutation_dominant_years[mut]:
                            mutation_dominant_years[mut] = year                            
    
    # 获取历史已流行突变列表
    dominant_mutations = set(mutation_dominant_years.keys())
    
    # === 为每种模型进行预测 ===
    for model_type in models:
        print(f"使用{model_type}模型进行预测...")
        
        # 初始化当前模型的风险突变列表
        risk_muts = []
        
        # 存储预测年份所有位点的预测流行度
        pred_prev = np.zeros(len(historical_data.columns))
        deltas = np.zeros(len(historical_data.columns))
        
        # === 遍历每个位点进行预测 ===
        for col_idx, col in enumerate(historical_data.columns):
            # 获取历史流行度数据
            freqs = historical_data[col].values
            
            # 确保有足够数据进行预测
            if len(freqs) < 2:
                pred_prev[col_idx] = freqs[-1] if len(freqs) > 0 else 0
                deltas[col_idx] = 0
                continue
                
            try:
                # 获取最近的流行度值
                last_value = freqs[-1]
                prev_value = freqs[-2]
                
                # 根据选择的模型预测流行度
                if model_type == 'linear':
                    # 多项式/线性模型: X_{t+1} = X_t + f_i
                    delta = last_value - prev_value  # 变化量
                    predicted = last_value + delta  # 简单线性预测
                    deltas[col_idx] = delta
                
                elif model_type == 'exponential':
                    # 指数模型: X_{t+1} = X_t * e^{f_i}
                    # 计算增长率 r = ln(X_t/X_{t-1})
                    if prev_value > 0.0001 and last_value > 0.0001:  # 避免对接近0的值取对数
                        growth_rate = np.log(last_value / prev_value)
                        predicted = last_value * np.exp(growth_rate)
                    else:
                        # 如果有接近0的值，退回到线性预测
                        delta = last_value - prev_value
                        predicted = last_value + delta
                
                elif model_type == 'logistic':
                    # Logistic模型: X_{t+1} = (X_t * e^{f_i}) / (X_t * e^{f_i} + (1-X_t))
                    # 计算logistic增长参数
                    if 0.0001 < prev_value < 0.9999 and 0.0001 < last_value < 0.9999:
                        # 求解f_i
                        f_i = np.log((last_value / (1 - last_value)) / (prev_value / (1 - prev_value)))
                        # 应用logistic函数预测
                        exp_term = np.exp(f_i)
                        predicted = (last_value * exp_term) / (last_value * exp_term + (1 - last_value))
                    else:
                        # 对于边界情况，使用线性预测
                        delta = last_value - prev_value
                        predicted = last_value + delta
                
                elif model_type == 'gompertz':
                    # 改进的Gompertz模型
                    if len(freqs) >= 3 and all(f > 0.0001 for f in freqs[-3:]):
                        try:
                            # 使用前三个点估计Gompertz参数
                            x0, x1, x2 = freqs[-3], freqs[-2], freqs[-1]
                            
                            # 计算两个增长比率
                            r1 = np.log(x1/x0)
                            r2 = np.log(x2/x1)
                            
                            # 计算增长减缓率
                            if abs(r1) > 0.0001:  # 避免除以接近0的数
                                decay = r2/r1
                                
                                # 预测下一个增长率
                                r3 = r2 * decay
                                
                                # 应用预测的增长率
                                predicted = x2 * np.exp(r3)
                            else:
                                # 如果第一个增长率接近0，使用简单指数
                                predicted = x2 * np.exp(r2)
                        except:
                            # 计算出错，退回到指数模型
                            growth_rate = np.log(last_value / prev_value)
                            predicted = last_value * np.exp(growth_rate)
                    else:
                        # 数据不足，退回到指数模型
                        if prev_value > 0.0001 and last_value > 0.0001:
                            growth_rate = np.log(last_value / prev_value)
                            predicted = last_value * np.exp(growth_rate)
                        else:
                            delta = last_value - prev_value
                            predicted = last_value + delta
                                # 确保预测值在[0,1]范围内
                pred_prev[col_idx] = np.clip(predicted, 0, 1)
            except Exception as e:
                print(f"预测{col}时出错({model_type}模型): {e}")
                # 发生异常时，使用最后一个已知值
                pred_prev[col_idx] = freqs[-1] if len(freqs) > 0 else 0
                deltas[col_idx] = 0  # 修改：异常情况下delta为0

        # === 风险突变判断 ===
        for col_idx, col in enumerate(historical_data.columns):
            pred = pred_prev[col_idx]
            
            # 格式化位点名称（去除X前缀）
            formatted_mut = col[1:] if col.startswith('X') else col
            
            # === 确定历史判断范围 ===
            # 检查该突变是否曾经成为过dominant mutation
            if formatted_mut in mutation_dominant_years:
                # 如果成为过dominant mutation，从成为dominant mutation的最晚年份之后开始判断
                start_year = mutation_dominant_years[formatted_mut]
                # 获取对应年份在historical_data中的索引
                historical_years = prev_data[prev_data.index < predict_season].index.tolist()
                try:
                    start_idx = historical_years.index(start_year)
                    freqs = historical_data[col].values[start_idx:]  # 从成为dominant mutation的最晚年份之后开始
                except ValueError:
                    # 如果找不到对应年份，使用全部历史数据
                    freqs = historical_data[col].values
            else:
                # 如果从未成为过dominant mutation，使用全部历史数据
                freqs = historical_data[col].values
            
            # === 突变条件判断 ===
            # 条件1：预测值≥theta且在相应历史范围内存在低流行（<theta）
            condition1 = (pred >= theta) and np.any(freqs < theta)
            
            # 条件2：预测值≥theta/10且在相应历史范围内存在极低流行（<theta/10）
            # 或者预测值≥theta*10且在相应历史范围内存在低流行（<theta*10）
            condition2 = False
            if theta/10 >= 0.01:
                condition2 = (pred >= theta/10) and np.any(freqs < theta/10)
            elif theta*10 < 1:
                condition2 = (pred >= theta*10) and np.any(freqs < theta*10)
            
            # 条件3：当前流感季的流行度<0.75（排除流行度≥0.75的突变）
            current_prev = historical_data[col].iloc[-1] if len(historical_data[col]) > 0 else 0
            condition3 = current_prev < 0.75
            
            if (condition1 or condition2) and condition3:
                risk_muts.append(col)
        
        # === 生成当前模型的预测记录 ===
        model_records = []
        for mut in risk_muts:
            # 格式化位点名称（去除X前缀）
            formatted_mut = mut[1:] if mut.startswith('X') else mut
            
            # 安全获取前一年流行度
            prev_prev = historical_data[mut].iloc[-1] if len(historical_data[mut]) > 0 else 0
                
            # 获取预测的流行度
            mut_idx = list(historical_data.columns).index(mut)
            predicted_prev = pred_prev[mut_idx]
            delta_val = deltas[mut_idx]
                
            model_records.append({
                'predict_season': predict_season,
                'risk_mutation': formatted_mut,
                'previous_prevalence': round(prev_prev, 4),
                'predicted_prevalence': round(predicted_prev, 4),
                'delta': round(delta_val, 4) if model_type == 'linear' else None,  # 只有linear模型才有delta值
                'model': model_type
            })
        
        # === 将当前模型结果添加到总结果中 ===
        if model_records:
            current_predictions = pd.DataFrame(model_records)
            all_predictions = pd.concat([all_predictions, current_predictions], ignore_index=True)
            
            print(f"{model_type}模型预测发现{len(model_records)}个风险突变")
        else:
            print(f"{model_type}模型未预测到任何风险突变")
    
    # 按突变名称排序，方便比较不同模型对相同突变的预测
    if not all_predictions.empty:
        all_predictions = all_predictions.sort_values(by=['risk_mutation', 'model'])
    
    # 如果指定了输出文件，保存结果
    if output_file is not None:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存所有预测结果到单一文件
        all_predictions.to_csv(output_file, index=False)
        print(f"已将所有模型预测结果保存至: {output_file}")
    
    print("所有模型预测完成!")
    return all_predictions

def analyze_risk_mutations(sequences_df, mutations_df, predict_season, semisphere, model=None):
    """
    分析风险突变在序列中的分布情况，包括处理缺失突变
    
    Args:
        sequences_df (DataFrame): 序列数据，包含序列信息和日期信息
        mutations_df (DataFrame): 突变预测数据，包含'risk_mutation'和'model'列
        predict_season (int): 预测年份
        semisphere (str): 'North' 或 'South'，用于确定时间范围
        model (str, optional): 要分析的特定模型名称。如果为None，分析所有模型
        
    Returns:
        DataFrame: 包含三列的数据框：
            - risk_mutation_group: 风险突变组合
            - count: 该组合在序列中出现的次数
            - model: 使用的预测模型（如果指定了model参数则所有行该值相同）
    """
    # 设置时间范围
    if semisphere == "North":
        start = f"{predict_season-1}-09-01"
        end = f"{predict_season}-02-01"
        submission_end = f"{predict_season}-02-01"
    else:
        start = f"{predict_season-1}-02-01"
        end = f"{predict_season-1}-09-01"
        submission_end = f"{predict_season-1}-09-01"

    # 确保日期格式正确
    sequences_df['collection_date'] = pd.to_datetime(sequences_df['collection_date'])
    sequences_df['submission_date'] = pd.to_datetime(sequences_df['submission_date'])

    # 筛选符合时间条件的序列
    date_mask = (
        (sequences_df['collection_date'] >= start) &
        (sequences_df['collection_date'] < end) &
        (sequences_df['submission_date'] < submission_end)
    )
    filtered_sequences = sequences_df[date_mask]
    
    # 根据model参数筛选突变数据
    if model is not None:
        # 如果指定了模型，只分析该模型的预测结果
        if 'model' in mutations_df.columns:
            mutations_df = mutations_df[mutations_df['model'] == model]
        else:
            print(f"警告: 突变数据中没有'model'列，将使用所有突变数据")

    # 如果mutations_df为空，返回空结果
    if mutations_df.empty:
        return pd.DataFrame(columns=['risk_mutation_group', 'count', 'model'])

    # 解析突变格式（包括处理缺失突变）
    def parse_mutation(mutation_str):
        position = int(''.join(filter(str.isdigit, mutation_str)))
        amino_acid = mutation_str.replace(str(position), '')  # 这样可以保留'-'符号
        return position, amino_acid

    # 按模型分组进行分析
    result_dfs = []
    
    # 确定要分析的模型
    if model is not None:
        models_to_analyze = [model]
    elif 'model' in mutations_df.columns:
        models_to_analyze = mutations_df['model'].unique()
    else:
        # 如果数据中没有model列，则视为单一模型
        models_to_analyze = ['undefined']
        
    # 为每个模型执行分析
    for current_model in models_to_analyze:
        # 筛选当前模型的突变
        if current_model != 'undefined' and 'model' in mutations_df.columns:
            model_mutations = mutations_df[mutations_df['model'] == current_model]
        else:
            model_mutations = mutations_df.copy()
            
        # 获取当前模型的所有风险突变的位置和氨基酸
        risk_mutations = [parse_mutation(mut) for mut in model_mutations['risk_mutation']]
        
        if not risk_mutations:
            continue  # 跳过没有风险突变的模型
        
        # 创建序列的副本用于当前模型的分析
        model_sequences = filtered_sequences.copy()

        # 在序列中查找风险突变组合
        def find_risk_mutations(row):
            mutations_found = []
            for pos, aa in risk_mutations:
                column_name = f'X{pos}'
                if column_name in row:
                    if aa == '-':  # 处理缺失突变
                        if pd.isna(row[column_name]) or row[column_name] == '-':
                            mutations_found.append(f"{pos}{aa}")
                    else:  # 处理普通突变
                        if row[column_name] == aa:
                            mutations_found.append(f"{pos}{aa}")
            return ','.join(sorted(mutations_found)) if mutations_found else None

        # 为每个序列标记风险突变组合
        model_sequences['risk_mutation_group'] = model_sequences.apply(find_risk_mutations, axis=1)
        
        # 移除没有风险突变的序列
        model_sequences = model_sequences[model_sequences['risk_mutation_group'].notna()]

        # 统计每个突变组合的出现次数
        if not model_sequences.empty:
            mutation_counts = model_sequences['risk_mutation_group'].value_counts().reset_index()
            mutation_counts.columns = ['risk_mutation_group', 'count']
            # 添加模型信息
            mutation_counts['model'] = current_model
            result_dfs.append(mutation_counts)
    
    # 合并所有模型的结果
    if result_dfs:
        return pd.concat(result_dfs, ignore_index=True)
    else:
        # 返回空DataFrame但保持列结构
        return pd.DataFrame(columns=['risk_mutation_group', 'count', 'model'])
    
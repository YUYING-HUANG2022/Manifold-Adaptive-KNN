# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 21:35:30 2026

@author: Meimei.Huang

初始化种群的相似度改用马氏距离，适应度中的相似度改用马氏距离
初始化策略：
    按类别构建候选集：每个类别选择 multiplier * region_size 个最近样本
    预计算_precompute_rank_scores
    初始化种群选择方法：基于概率选择（概率基于相对排名得分）
    样本代表性评估：综合考虑样本能否代表其类别，以及类别对测试点的接受程度 _calculate_sample_representativeness
    相对排名得分计算：基于类别均衡的代表性评估 _calculate_relative_rank_score
    调整因子计算：基于代表性评分的距离惩罚调整  _calculate_adjustment_factor
种群个体相似度指标：
    保留_calculate_similarity_old
    增加_calculate_similarity：基于代表性得分的调整因子*马氏距离
使用wineQualityN数据库 
质心法：
传统数学分析+局部协方差流形约束：

广义瑞利商+全局广义瑞利商流形约束：
正确率:0.8787878787878788  （_calculate_similarity：反向排名指标）
正确率:0.8181818181818182
使用iris数据库
广义瑞利商+全局广义瑞利商流形约束：


"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from collections import Counter
from sklearn.base import BaseEstimator
from scipy.linalg import logm, fractional_matrix_power
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.utils.validation import check_X_y
import random
# ==================== 改进的遗传算法优化器 ====================
class ImprovedGeneticAlgorithmOptimizer:
    """改进的遗传算法优化器，解决类别不平衡问题"""
    def __init__(self, population_size=30, generations=50, crossover_rate=0.9,
                 mutation_rate=0.1, elite_size=6, inv_cov_matrix=None):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.inv_cov_matrix = inv_cov_matrix
        # 预计算存储结构
        self.forward_ranks = {} # 样本索引 -> 向前排名
        self.backward_ranks = {} # 样本索引 -> 向后排名  
        self.relative_rank_scores = {} # 样本索引 -> 相对排名得分
        self.rank_adjustment_factors = {} # 样本索引 -> 排名调整因子

    def initialize_population(self, data_size, region_size, X, y, target_point):
        """
        改进的种群初始化策略：预计算候选样本的双向排名和相对排名得分
        """
        # 1. 按类别构建候选集
        multiplier = 2
        distances = self._mahalanobis_distance(X, target_point)
        distances = np.array(distances)
        unique_classes = np.unique(y)
        # 为每个类别独立选择最近的样本
        class_candidates = {}
        candidate_indices = []
        for cls in unique_classes:
            class_indices = np.where(y == cls)[0]
            if len(class_indices) == 0:
                continue
            class_distances = distances[class_indices]
            n_to_select = min(multiplier * region_size, len(class_indices))
            closest_indices = class_indices[np.argsort(class_distances)[:n_to_select]]
            class_candidates[cls] = closest_indices.tolist()
            candidate_indices.extend(closest_indices)
            print(f"类别 {cls}: 选择了 {len(closest_indices)} 个最近样本")
        candidate_indices = np.array(candidate_indices)
        self.candidate_indices = candidate_indices
        self.candidate_set = set(candidate_indices)
        self.class_candidates = class_candidates
        # 2. 预计算每个候选样本的双向排名和相对排名得分
        self._precompute_rank_scores(X, y, target_point, class_candidates)
        print(f"候选集总大小: {len(candidate_indices)}，包含类别分布: {dict(Counter(y[candidate_indices]))}")
        # 3. 计算每个类别在个体中应有的样本数
        global_class_dist = Counter(y)
        total_global = sum(global_class_dist.values())
        class_targets = {}
        for cls in unique_classes:
            target_count = max(1, int(region_size * global_class_dist[cls] / total_global))
            available_count = len(class_candidates.get(cls, []))
            class_targets[cls] = min(target_count, available_count)
        # 调整目标样本数，确保总和为region_size
        total_target = sum(class_targets.values())
        if total_target != region_size:
            diff = region_size - total_target
            sorted_classes = sorted(class_targets.keys(),
                                  key=lambda x: len(class_candidates.get(x, [])), 
                                  reverse=diff > 0)
            for cls in sorted_classes:
                if diff == 0:
                    break
                if diff > 0 and len(class_candidates.get(cls, [])) > class_targets[cls]:
                    class_targets[cls] += 1
                    diff -= 1
                elif diff < 0 and class_targets[cls] > 1:
                    class_targets[cls] -= 1
                    diff += 1
        # 保存类别目标样本数到self中
        self.class_targets = class_targets
        print(f"各类别目标样本数: {class_targets}")
        # 4. 初始化种群（基于排名得分优化选择）
        population = self._initialize_population_with_rank_scores(
            region_size, X, y, target_point, class_candidates, distances
        )
        return population
    def _precompute_rank_scores(self, X, y, target_point, class_candidates):
        """
        修正的预计算函数：使用新的代表性计算方法
        """
        print("开始预计算候选样本的代表性得分...")
        # 清空存储
        self.forward_ranks.clear()
        self.backward_ranks.clear()
        self.relative_rank_scores.clear()
        self.rank_adjustment_factors.clear()
        for cls, candidates in class_candidates.items():
            if len(candidates) == 0:
                continue
            candidate_set_size = len(candidates)
            for sample_idx in candidates:
                # 计算排名
                forward_rank = self._calculate_forward_rank(
                    sample_idx, target_point, candidates, X, cls
                )
                backward_rank = self._calculate_backward_rank(
                    sample_idx, target_point, candidates, X, cls
                )
                # 存储原始排名
                self.forward_ranks[sample_idx] = forward_rank
                self.backward_ranks[sample_idx] = backward_rank
                # 计算相对排名得分（使用新的代表性计算方法）
                relative_score = self._calculate_relative_rank_score(
                    forward_rank, backward_rank, candidate_set_size
                )
                self.relative_rank_scores[sample_idx] = relative_score
                # 计算调整因子
                adjustment_factor = self._calculate_adjustment_factor(
                    forward_rank, backward_rank, candidate_set_size
                )
                self.rank_adjustment_factors[sample_idx] = adjustment_factor
        print(f"代表性得分预计算完成，共处理 {sum(len(c) for c in class_candidates.values())} 个样本")
    def _calculate_sample_representativeness(self, forward_rank, backward_rank, candidate_set_size):
        """
        修正的样本代表性评估：区分类别代表性和类别接受度
        综合考虑样本能否代表其类别，以及类别对测试点的接受程度
        """
        if candidate_set_size <= 1:
            return 0.5 # 候选集太小，无法准确评估
        # 计算相对排名位置（0-1范围，1表示最优）
        forward_relative = 1 - (forward_rank - 1) / (candidate_set_size - 1)
        backward_relative = 1 - (backward_rank - 1) / (candidate_set_size - 1)
        # 1. 计算类别代表性：样本在类别中的典型程度
        # 向前排名靠前（forward_relative高）说明样本是类别的典型代表
        class_representativeness = forward_relative
        # 2. 计算类别接受度：类别对测试点的欢迎程度  
        # 向后排名靠前（backward_relative高）说明类别对测试点接受度高
        class_acceptance = backward_relative
        # 3. 检测特异点情况
        is_typical_outlier = (forward_relative > 0.7) and (backward_relative < 0.3)
        is_receptive_outlier = (forward_relative < 0.3) and (backward_relative > 0.7)
        # 4. 综合代表性评分（基于您提到的光谱多元建模中的代表性样本选择原理[4](@ref)）
        if is_typical_outlier:
            # 情况1：样本是类别典型但类别不接受测试点 - 代表性较低
            representativeness_score = 0.3 * class_representativeness + 0.7 * class_acceptance
        elif is_receptive_outlier:
            # 情况2：类别接受测试点但样本非典型 - 中等代表性
            representativeness_score = 0.5 * class_representativeness + 0.5 * class_acceptance
        else:
            # 情况3：样本典型且类别接受 - 高代表性
            representativeness_score = 0.4 * class_representativeness + 0.6 * class_acceptance
        # 5. 应用平滑处理，确保评分在0-1范围内
        final_score = max(0.0, min(1.0, representativeness_score))
        return final_score
    def _calculate_relative_rank_score(self, forward_rank, backward_rank, candidate_set_size):
        """
        修正的相对排名得分计算：基于类别均衡的代表性评估
        """
        # 计算样本代表性
        representativeness = self._calculate_sample_representativeness(
            forward_rank, backward_rank, candidate_set_size
        )
        # 根据代表性程度划分评分等级
        if representativeness > 0.8:
            return 0.9 # 高代表性
        elif representativeness > 0.6:
            return 0.7 # 中等代表性
        elif representativeness > 0.4:
            return 0.5 # 一般代表性
        elif representativeness > 0.2:
            return 0.3 # 低代表性
        else:
            return 0.1 # 极低代表性
    def _calculate_forward_rank(self, sample_idx, target_point, cls_candidates, X, cls):
        """
        计算正向排名：在样本类别的候选集中，该样本与测试点的距离排名
        """
        # 获取候选集中所有样本到测试点的距离
        candidate_points = X[cls_candidates]
        candidate_distances = self._mahalanobis_distance(candidate_points, target_point)
        # 计算当前样本到测试点的距离
        sample_point = X[sample_idx]
        sample_distance = self._mahalanobis_distance(sample_point, target_point)
        # 将所有距离合并排序（包括当前样本）
        all_distances = np.append(candidate_distances, sample_distance)
        # 排序并获取排名（距离越小排名越高，排名从1开始）
        sorted_indices = np.argsort(all_distances)
        rank = np.where(sorted_indices == len(all_distances) - 1)[0][0] + 1
        return rank
    def _calculate_backward_rank(self, sample_idx, target_point, cls_candidates, X, cls):
        """
        计算反向排名：以当前样本为参考点，测试点在候选集中的距离排名
        """
        # 获取当前样本的特征向量
        sample_point = X[sample_idx]
        # 计算候选集中所有样本到当前样本的距离
        candidate_points = X[cls_candidates]
        candidate_distances = self._mahalanobis_distance(candidate_points, sample_point)
        # 计算测试点到当前样本的距离
        target_distance = self._mahalanobis_distance(target_point, sample_point)
        # 将所有距离合并排序（包括测试点）
        all_distances = np.append(candidate_distances, target_distance)
        # 排序并获取排名（距离越小排名越高，排名从1开始）
        sorted_indices = np.argsort(all_distances)
        rank = np.where(sorted_indices == len(all_distances) - 1)[0][0] + 1
        return rank
    def _compute_adjustment_factor_for_sample(self, sample_idx, X, y, target_point):
        """
        动态计算单个样本的调整因子（当样本未预计算时）
        """
        cls = y[sample_idx]
        # 检查类别是否存在候选集中
        if not hasattr(self, 'class_candidates') or cls not in self.class_candidates:
            return 1.0 # 默认中性调整因子
        cls_candidates = self.class_candidates[cls]
        if len(cls_candidates) == 0:
            return 1.0
        # 计算向前排名和向后排名
        forward_rank = self._calculate_forward_rank(sample_idx, target_point, cls_candidates, X, cls)
        backward_rank = self._calculate_backward_rank(sample_idx, target_point, cls_candidates, X, cls)
        # 计算调整因子
        adjustment_factor = self._calculate_adjustment_factor(forward_rank, backward_rank)
        # 存储到预计算字典以供后续使用
        self.rank_adjustment_factors[sample_idx] = adjustment_factor
        return adjustment_factor
    def _calculate_adjustment_factor(self, forward_rank, backward_rank, candidate_set_size):
        """
        修正的调整因子计算：基于代表性评分的距离惩罚调整
        """
        # 获取相对排名得分
        acceptance_score = self._calculate_relative_rank_score(
            forward_rank, backward_rank, candidate_set_size
        )
        # 根据代表性得分调整距离惩罚
        # 代表性越高，距离惩罚越小（调整因子越小）
        if acceptance_score > 0.8: # 高代表性
            adjustment_factor = 0.5
        elif acceptance_score > 0.6: # 中等代表性
            adjustment_factor = 0.7
        elif acceptance_score > 0.4: # 一般代表性
            adjustment_factor = 1.0
        elif acceptance_score > 0.2: # 低代表性
            adjustment_factor = 1.3
        else: # 极低代表性
            adjustment_factor = 1.6
        return adjustment_factor
    def _initialize_population_with_rank_scores(self, region_size, X, y, target_point, class_candidates, distances):
        """
        改进的种群初始化：引入随机性和多样性，避免所有个体相同
        使用概率选择而非确定性选择，确保每个个体都有不同的样本组合
        """
        population = []
        unique_classes = np.unique(y)
        # 预计算所有候选样本的相对排名得分（如果尚未计算）
        # self._ensure_all_candidate_scores_computed(X, y, target_point, class_candidates)
        for ind_idx in range(self.population_size):
            individual = []
            used_samples_local = set()
            # 策略1: 按类别比例选择，但使用概率选择而非确定性选择
            for cls in unique_classes:
                if cls not in class_candidates:
                    continue
                target_count = self.class_targets[cls]
                if target_count <= 0:
                    continue
                # 获取该类别的候选样本及其相对排名得分
                cls_candidates_list = class_candidates[cls]
                # 如果候选样本数量不足，直接选择所有样本
                if len(cls_candidates_list) <= target_count:
                    for sample_idx in cls_candidates_list:
                        if sample_idx not in used_samples_local and len(individual) < region_size:
                            individual.append(sample_idx)
                            used_samples_local.add(sample_idx)
                    continue
                # 计算选择概率（基于相对排名得分）
                candidate_scores = []
                valid_candidates = []
                for sample_idx in cls_candidates_list:
                    if sample_idx in self.relative_rank_scores:
                        score = self.relative_rank_scores[sample_idx]
                        candidate_scores.append(score)
                        valid_candidates.append(sample_idx)
                # 如果所有样本得分相同，使用均匀分布
                if len(set(candidate_scores)) == 1:
                    # 所有得分相同，使用均匀随机选择
                    selected_indices = np.random.choice(
                        len(valid_candidates), 
                        size=min(target_count, len(valid_candidates)), 
                        replace=False
                    )
                else:
                    # 使用得分作为权重进行概率选择
                    # 对得分进行softmax处理，转换为概率
                    scores_array = np.array(candidate_scores)
                    exp_scores = np.exp(scores_array - np.max(scores_array)) # 防止数值溢出
                    probabilities = exp_scores / np.sum(exp_scores)
                    # 概率选择（避免重复）
                    selected_indices = np.random.choice(
                        len(valid_candidates), 
                        size=min(target_count, len(valid_candidates)), 
                        replace=False, 
                        p=probabilities
                    )
                # 添加选中的样本
                for idx in selected_indices:
                    sample_idx = valid_candidates[idx]
                    if sample_idx not in used_samples_local and len(individual) < region_size:
                        individual.append(sample_idx)
                        used_samples_local.add(sample_idx)
            # 策略2: 如果样本不足，使用多样性策略补充
            if len(individual) < region_size:
                remaining = region_size - len(individual)
                self._diversity_based_supplement(individual, used_samples_local, remaining, X, y)
            # 策略3: 如果样本过多，使用多样性策略删除
            if len(individual) > region_size:
                to_remove = len(individual) - region_size
                self._diversity_based_removal(individual, to_remove, X, y)
            # 最终确保个体大小正确
            if len(individual) != region_size:
                individual = self._force_adjust_size_diversity(
                    individual, region_size, X, y, used_samples_local
                )
            population.append(individual)
            # 调试信息（可选）
            if ind_idx < 3: # 只打印前3个个体信息避免输出过多
                individual_scores = [self.relative_rank_scores.get(idx, 0.5) for idx in individual]
                print(f"个体 {ind_idx}: 大小 {len(individual)}, 平均得分: {np.mean(individual_scores):.3f}, 样本: {individual[:5]}...") # 只显示前5个样本
        print(f"种群初始化完成，共 {len(population)} 个个体")
        # 验证种群多样性
        self._validate_population_diversity(population)
        return population
    def _diversity_based_supplement(self, individual, used_samples_local, remaining, X, y):
        """
        基于多样性的样本补充策略
        考虑类别平衡和样本多样性
        """
        if remaining <= 0:
            return
        # 获取所有可用的候选样本
        available_candidates = list(self.candidate_set - used_samples_local)
        if not available_candidates:
            return
        # 计算当前个体的类别分布
        current_class_dist = Counter([y[idx] for idx in individual])
        # 计算需要补充的类别分布（基于全局目标）
        target_class_dist = self.class_targets
        supplement_plan = {}
        for cls, target_count in target_class_dist.items():
            current_count = current_class_dist.get(cls, 0)
            if current_count < target_count:
                supplement_plan[cls] = min(target_count - current_count, remaining)
        # 如果按类别补充后仍有剩余，随机分配
        total_planned = sum(supplement_plan.values())
        if total_planned < remaining:
            # 将剩余名额分配给样本最少的类别
            remaining_slots = remaining - total_planned
            for _ in range(remaining_slots):
                # 找到当前比例最低的类别
                min_ratio_cls = min(target_class_dist.keys(), 
                                   key=lambda cls: current_class_dist.get(cls, 0) / target_class_dist[cls])
                supplement_plan[min_ratio_cls] = supplement_plan.get(min_ratio_cls, 0) + 1
        # 执行补充
        for cls, count in supplement_plan.items():
            if count <= 0:
                continue
            # 获取该类别可用的候选样本
            cls_available = [idx for idx in available_candidates if y[idx] == cls]
            if not cls_available:
                continue
            # 计算选择概率（基于得分）
            cls_scores = [self.relative_rank_scores.get(idx, 0.5) for idx in cls_available]
            if len(set(cls_scores)) == 1:
                # 得分相同，随机选择
                selected = np.random.choice(cls_available, size=min(count, len(cls_available)), replace=False)
            else:
                # 概率选择
                scores_array = np.array(cls_scores)
                exp_scores = np.exp(scores_array - np.max(scores_array))
                probabilities = exp_scores / np.sum(exp_scores)
                selected = np.random.choice(
                    cls_available, 
                    size=min(count, len(cls_available)), 
                    replace=False, 
                    p=probabilities
                )
            for sample_idx in selected:
                if len(individual) >= len(individual) + remaining: # 检查是否已满
                    break
                if sample_idx not in used_samples_local:
                    individual.append(sample_idx)
                    used_samples_local.add(sample_idx)
    def _diversity_based_removal(self, individual, to_remove, X, y):
        """
        基于多样性的样本删除策略
        优先删除冗余样本，保持类别平衡
        """
        if to_remove <= 0:
            return
        # 计算当前个体的类别分布
        current_class_dist = Counter([y[idx] for idx in individual])
        # 计算每个类别的超额数量
        excess_plan = {}
        for cls, current_count in current_class_dist.items():
            target_count = self.class_targets.get(cls, 0)
            excess = current_count - target_count
            if excess > 0:
                excess_plan[cls] = min(excess, to_remove)
        # 如果按类别删除后仍有剩余，从比例最高的类别中删除
        total_planned = sum(excess_plan.values())
        if total_planned < to_remove:
            remaining_remove = to_remove - total_planned
            # 从当前比例最高的类别中删除
            sorted_classes = sorted(current_class_dist.keys(),
                                  key=lambda cls: current_class_dist[cls] / self.class_targets.get(cls, 1),
                                  reverse=True)
            for cls in sorted_classes:
                if remaining_remove <= 0:
                    break
                if cls not in excess_plan:
                    excess_plan[cls] = 0
                remove_count = min(remaining_remove, current_class_dist[cls])
                excess_plan[cls] += remove_count
                remaining_remove -= remove_count
        # 执行删除
        for cls, remove_count in excess_plan.items():
            if remove_count <= 0:
                continue
            # 获取该类别的样本索引
            cls_indices = [idx for idx in individual if y[idx] == cls]
            if not cls_indices:
                continue
            # 计算删除概率（得分低的样本更容易被删除）
            cls_scores = [self.relative_rank_scores.get(idx, 0.5) for idx in cls_indices]
            if len(set(cls_scores)) == 1:
                # 得分相同，随机删除
                to_delete = np.random.choice(cls_indices, size=min(remove_count, len(cls_indices)), replace=False)
            else:
                # 得分低的样本更容易被删除
                scores_array = np.array(cls_scores)
                # 将得分转换为删除概率（得分越低，删除概率越高）
                delete_probs = 1 - (scores_array / np.max(scores_array))
                delete_probs = delete_probs / np.sum(delete_probs) # 归一化
                to_delete = np.random.choice(
                    cls_indices,
                    size=min(remove_count, len(cls_indices)),
                    replace=False,
                    p=delete_probs
                )
            # 从个体中删除选中的样本
            for sample_idx in to_delete:
                if sample_idx in individual:
                    individual.remove(sample_idx)
    def _force_adjust_size_diversity(self, individual, target_size, X, y, used_samples_local):
        """
        基于多样性的强制大小调整
        """
        current_size = len(individual)
        if current_size > target_size:
            # 删除多余的样本，保持多样性
            to_remove = current_size - target_size
            self._diversity_based_removal(individual, to_remove, X, y)
        else:
            # 补充样本，保持多样性
            to_add = target_size - current_size
            self._diversity_based_supplement(individual, used_samples_local, to_add, X, y)
        # 最终检查
        if len(individual) > target_size:
            individual = individual[:target_size]
        elif len(individual) < target_size:
            # 紧急补充：随机选择可用样本
            remaining = target_size - len(individual)
            available = list(self.candidate_set - set(individual))
            if available:
                additional = np.random.choice(available, size=min(remaining, len(available)), replace=False)
                individual.extend(additional)
        return individual[:target_size] # 确保大小正确
    def _validate_population_diversity(self, population):
        """
        验证种群多样性
        """
        if len(population) <= 1:
            return
        # 计算个体间的相似度
        unique_individuals = set()
        for individual in population:
            # 将个体转换为可哈希的元组
            individual_tuple = tuple(sorted(individual))
            unique_individuals.add(individual_tuple)
        diversity_ratio = len(unique_individuals) / len(population)
        print(f"种群多样性验证: 唯一个体数 {len(unique_individuals)}/{len(population)}, 多样性比率: {diversity_ratio:.3f}")
        if diversity_ratio < 0.8:
            print("警告: 种群多样性较低，考虑增加随机性")
    def calculate_class_discriminability(self, X_region, y_region):
        """
        使用LDA评估整体类别可区分性
        返回LDA模型的解释方差比例作为可区分性度量
        """
        if len(X_region) < 2 or len(np.unique(y_region)) < 2:
            return 0.0
        
        try:
            # 使用LDA评估类别可区分性
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_region, y_region)
            
            # 使用LDA的解释方差比例作为可区分性度量
            if hasattr(lda, 'explained_variance_ratio_'):
                # 解释方差比例之和反映了LDA模型的判别能力
                discriminability = np.sum(lda.explained_variance_ratio_)
            else:
                # 备选方案：计算LDA投影后的类别分离度
                if hasattr(lda, 'transform'):
                    X_projected = lda.transform(X_region)
                    # 计算投影后的类别间方差与类别内方差的比值
                    unique_classes = np.unique(y_region)
                    overall_mean = np.mean(X_projected, axis=0)
                    between_var = 0
                    within_var = 0
                    for cls in unique_classes:
                        cls_indices = y_region == cls
                        cls_points = X_projected[cls_indices]
                        cls_mean = np.mean(cls_points, axis=0)
                        cls_size = len(cls_points)
                        # 类别间方差
                        between_var += cls_size * np.sum((cls_mean - overall_mean) ** 2)
                        # 类别内方差
                        within_var += np.sum((cls_points - cls_mean) ** 2)    
                    if within_var > 0:
                        discriminability = between_var / within_var
                        # 归一化到0-1范围
                        discriminability = min(1.0, discriminability / 10.0)  # 经验值缩放
                    else:
                        discriminability = 0.0
                else:
                    discriminability = 0.0
            return min(1.0, max(0.0, discriminability))
        
        except Exception as e:
            # LDA可能因为各种原因失败（如奇异矩阵）
            print(f"LDA计算失败: {e}")
        return 0.0
    def calculate_feature_discriminative_power(self, X_region, y_region):
        """
        计算每个特征的Fisher判别得分
        专门用于特征选择和权重计算
        """
        n_features = X_region.shape[1]
        unique_classes = np.unique(y_region)
        if len(unique_classes) < 2:
            return np.zeros(n_features)
        # 计算每个特征的Fisher得分
        fisher_scores = []
        for feature_idx in range(n_features):
            feature_values = X_region[:, feature_idx] 
            # 计算总体均值
            overall_mean = np.mean(feature_values)
            # 计算类别间方差（Between-class variance）
            between_var = 0
            total_samples = len(feature_values)
            for cls in unique_classes:
                cls_values = feature_values[y_region == cls]
                if len(cls_values) > 0:
                    cls_mean = np.mean(cls_values)
                    cls_weight = len(cls_values) / total_samples
                    between_var += cls_weight * (cls_mean - overall_mean) ** 2
            # 计算类别内方差（Within-class variance）
            within_var = 0
            for cls in unique_classes:
                cls_values = feature_values[y_region == cls]
                if len(cls_values) > 0:
                    within_var += np.var(cls_values) * len(cls_values) / total_samples
            # 计算Fisher得分
            if within_var > 0:
                fisher_score = between_var / within_var
            else:
                fisher_score = 0.0
            fisher_scores.append(fisher_score)
        return np.array(fisher_scores)
    def calculate_similarity(self, individual, X_region, y_region, target_point, X, y):
        """
        相似度计算：使用基于新代表性评估的调整因子
        """
        if len(individual) == 0:
            return 0.0
        # 计算原始距离
        distances = self._mahalanobis_distance(X_region, target_point)
        adjusted_distances = []
        for i, sample_idx in enumerate(individual):
            cls = y[sample_idx]
            # 获取候选集大小
            candidate_set_size = len(self.class_candidates.get(cls, []))
            if candidate_set_size == 0:
                # 没有候选集，使用原始距离
                adjusted_distances.append(distances[i])
                continue
            # 获取或计算调整因子（使用新的代表性评估逻辑）
            if sample_idx in self.rank_adjustment_factors:
                adjustment_factor = self.rank_adjustment_factors[sample_idx]
            else:
                # 动态计算
                if cls in self.class_candidates:
                    candidates = self.class_candidates[cls]
                    forward_rank = self._calculate_forward_rank(
                        sample_idx, target_point, candidates, X, cls
                    )
                    backward_rank = self._calculate_backward_rank(
                        sample_idx, target_point, candidates, X, cls
                    )
                    adjustment_factor = self._calculate_adjustment_factor(
                        forward_rank, backward_rank, candidate_set_size
                    )
                    self.rank_adjustment_factors[sample_idx] = adjustment_factor
                else:
                    adjustment_factor = 1.0 # 默认值
            # 应用调整
            adjusted_distance = distances[i] * adjustment_factor
            adjusted_distances.append(adjusted_distance)
        # 计算调整后的平均距离
        if len(adjusted_distances) == 0:
            return 0.0
        weighted_mean_distance = np.mean(adjusted_distances)
        return 1 / (1 + weighted_mean_distance)

    def calculate_similarity_tightness(self, X_region, y_region, target_point, X, y):
        """
        改进的相似度计算：考虑类别紧度
        给予紧度高的类别更高权重，提高相似度评估的可靠性
        """
        if len(X_region) == 0:
            return 0.0        
        # 1. 计算样本到目标点的原始距离
        distances = self._mahalanobis_distance(X_region, target_point)
        # 2. 根据类别紧度调整距离
        adjusted_distances = self._adjust_distances_by_tightness(
            distances, y_region, self.class_tightness
        )
        # 3. 计算加权平均距离
        weighted_mean_distance = np.mean(adjusted_distances)
        return 1 / (1 + weighted_mean_distance)
    def _calculate_class_tightness(self):
        """
        计算每个类别的紧度（类内样本的平均距离）
        紧度越高表示类别内样本越聚集
        """
        class_tightness = {}
        for cls in self.class_candidates.keys():
            # 获取当前类别的所有样本
            cls_samples = self.class_candidates[cls]
            if len(cls_samples) <= 1:
                # 如果类别只有一个样本，紧度设为1（最高）
                class_tightness[cls] = 1.0
                continue
            # 计算类内样本间的平均距离
            intra_class_distances = []
            # 方法1: 计算样本到类中心的平均距离
            class_center = np.mean(cls_samples, axis=0)
            center_distances = np.linalg.norm(cls_samples - class_center, axis=0)
            center_based_tightness = 1 / (1 + np.mean(center_distances))
            # 方法2: 计算样本间两两距离的平均值
            pairwise_distances = []
            for i in range(len(cls_samples)):
                for j in range(i + 1, len(cls_samples)):
                    distance = np.linalg.norm(cls_samples[i] - cls_samples[j])
                    pairwise_distances.append(distance)
            if pairwise_distances:
                pairwise_based_tightness = 1 / (1 + np.mean(pairwise_distances))
            else:
                pairwise_based_tightness = 1.0
            # 综合两种方法的紧度评估
            combined_tightness = 0.7 * center_based_tightness + 0.3 * pairwise_based_tightness
            class_tightness[cls] = min(1.0, max(0.1, combined_tightness)) # 限制在[0.1, 1.0]范围内
        return class_tightness
    def _adjust_distances_by_tightness(self, distances, y_region, class_tightness):
        """
        根据类别紧度调整距离
        紧度高的类别，距离惩罚较小；紧度低的类别，距离惩罚较大
        """
        adjusted_distances = np.copy(distances)
        for i, cls in enumerate(y_region):
            tightness = class_tightness.get(cls, 0.5) # 默认紧度为0.5
            # 紧度调整因子：紧度越高，调整因子越小（距离惩罚越小）
            # tightness ∈ [0.1, 1.0]，调整因子 ∈ [0.5, 2.0]
            adjustment_factor = 2.0 - 1.5 * tightness # 紧度1.0→调整因子0.5，紧度0.1→调整因子1.85
            adjusted_distances[i] = distances[i] * adjustment_factor
        return adjusted_distances
    def _mahalanobis_distance(self, point_set, target):
        """
        计算马氏距离（修复版）
        """
        # 确保输入是二维数组
        if point_set.ndim == 1:
            point_set = point_set.reshape(1, -1)
        if target.ndim == 1:
            target = target.reshape(1, -1)
        distances = []
        for i in range(len(point_set)):
            point = point_set[i].reshape(1, -1)
            diff = point - target
            # 计算马氏距离
            temp = np.dot(diff, self.inv_cov_matrix)
            mahalanobis_dist = np.sqrt(np.dot(temp, diff.T))[0, 0]
            distances.append(mahalanobis_dist)
        if len(distances) == 1:
            return distances[0]
        return np.array(distances)
    def calculate_balanced_purity(self, y_region, y):
        """
        平衡的纯度计算，考虑全局类别分布
        避免偏向多数类
        """
        if len(y_region) == 0:
            return 0.0
        # 计算局部区域类别分布
        local_class_dist = Counter(y_region)
        # 计算全局类别分布
        global_class_dist = Counter(y)
        total_global = sum(global_class_dist.values())
        # 计算每个类别的期望比例
        expected_proportions = {}
        for cls in global_class_dist:
            expected_proportions[cls] = global_class_dist[cls] / total_global
        # 计算实际比例与期望比例的差异
        purity_score = 0
        for cls in set(list(local_class_dist.keys()) + list(expected_proportions.keys())):
            actual_prop = local_class_dist.get(cls, 0) / len(y_region)
            expected_prop = expected_proportions.get(cls, 0)
            # 使用KL散度思想，但避免log(0)
            if actual_prop > 0 and expected_prop > 0:
                purity_score += actual_prop * np.log(actual_prop / expected_prop)
            elif actual_prop > 0:
                # 如果全局中没有这个类别，但局部区域有，惩罚这种情况
                purity_score -= 1
        # 转换为0-1范围的分数（越高越好）
        # 使用sigmoid函数将分数映射到0-1
        balanced_purity = 1 / (1 + np.exp(-purity_score))
        return balanced_purity
    def calculate_class_diversity(self, y_region, y):
        """
        计算类别多样性，鼓励包含多个类别
        但避免包含过多不相关的类别
        """
        if len(y_region) == 0:
            return 0.5
        unique_classes = len(set(y_region))
        total_classes = len(set(y))
        # 类别多样性 = 局部区域类别数 / 总类别数
        class_diversity = unique_classes / total_classes
        # 但我们需要适度的多样性，不是越多越好
        # 使用一个峰值在适度多样性处的函数
        # 假设最佳多样性是总类别数的一半
        optimal_diversity = 0.5 # 适度的多样性
        diversity_score = 1 - abs(class_diversity - optimal_diversity)
        return max(0, diversity_score)
    def evaluate_fitness(self, individual, X, y, target_point, target_class=None):
        """
        改进的适应度评估函数
        考虑类别平衡问题
        """
        if len(individual) == 0:
            return 0
        X_region = X[individual]
        y_region = y[individual]
        # 计算三个关键指标（改进版本）
        similarity = self.calculate_similarity(individual,X_region, y_region, target_point, X, y)
        purity = self.calculate_balanced_purity(y_region, y)
        diversity = self.calculate_class_diversity(y_region, y)
        lda_discriminability = self.calculate_class_discriminability(X_region, y_region)
        fisher_scores = self.calculate_feature_discriminative_power(X_region, y_region)
        avg_fisher_score = np.mean(fisher_scores) if len(fisher_scores) > 0 else 0
        normalize_avg_fisher_score = 1/(1+np.exp(-0.1*avg_fisher_score))
        # 如果知道目标类别，增加目标类别奖励
        target_class_bonus = 0
        if target_class is not None and target_class in y_region:
            target_count = list(y_region).count(target_class)
            target_ratio = target_count / len(y_region)
            # 目标类别奖励，但不过度强调
            target_class_bonus = min(0.2, target_ratio * 0.3)
        # 综合适应度（调整权重）
        fitness = (similarity * 0.6 + 
                  purity * 0.3 + 
                  
                  target_class_bonus)
        # 调试信息
        if random.random() < 0.05: # 5%概率打印调试信息
            class_distribution = Counter(y_region)
            # print(f"适应度分解: 相似度={similarity:.3f}, 纯度={purity:.3f}, 多样性={diversity:.3f}, 目标奖励={target_class_bonus:.3f}, 综合={fitness:.3f}")
            # print(f" 类别分布: {dict(class_distribution)}")
        return fitness
    def select_parents(self, population, fitness_scores):
        """轮盘赌选择父代"""
        # 确保适应度为正
        min_fitness = min(fitness_scores)
        if min_fitness < 0:
            fitness_scores = [f - min_fitness + 0.01 for f in fitness_scores]
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choices(population, k=2)
        probabilities = [fitness / total_fitness for fitness in fitness_scores]
        parents = random.choices(population, weights=probabilities, k=2)
        return parents
    def crossover(self, parent1, parent2, y, region_size):
        """
        改进的交叉操作：基于相对排名得分优化样本选择
        优先选择向后排名比向前排名更好的样本（相对排名得分高的样本）
        """
        if random.random() < self.crossover_rate:
            # 合并两个父代，去重
            combined = list(set(parent1 + parent2))
            # 如果没有预计算数据，回退到随机选择
            if not hasattr(self, 'relative_rank_scores') or not self.relative_rank_scores:
                child = random.sample(combined, min(region_size, len(combined)))
                if len(child) < region_size:
                    # 补充随机样本
                    remaining = region_size - len(child)
                    additional = random.choices(combined, k=remaining)
                    child.extend(additional)
                return child[:region_size]
            # 计算合并样本的相对排名得分
            sample_scores = {}
            for sample_idx in combined:
                if sample_idx in self.relative_rank_scores:
                    sample_scores[sample_idx] = self.relative_rank_scores[sample_idx]
                else:
                    # 动态计算相对排名得分
                    cls = y[sample_idx]
                    if hasattr(self, 'class_candidates') and cls in self.class_candidates:
                        cls_candidates = self.class_candidates[cls]
                        forward_rank = self._calculate_forward_rank(sample_idx, self.target_point, cls_candidates, self.X, cls)
                        backward_rank = self._calculate_backward_rank(sample_idx, self.target_point, cls_candidates, self.X, cls)
                        score = self._calculate_relative_rank_score(forward_rank, backward_rank)
                        self.relative_rank_scores[sample_idx] = score
                    else:
                        score = 0.5 # 默认值
                    sample_scores[sample_idx] = score
            # 按相对排名得分降序排序
            sorted_samples = sorted(combined, key=lambda x: sample_scores.get(x, 0.5), reverse=True)
            # 优先选择相对排名得分高的样本
            child = []
            used_classes = Counter()
            # 第一步：优先选择高得分样本，同时保持类别平衡
            for sample_idx in sorted_samples:
                if len(child) >= region_size:
                    break
                cls = y[sample_idx]
                target_count = self.class_targets.get(cls, 0)
                current_count = used_classes.get(cls, 0)
                # 如果该类别还有名额，且样本相对排名得分高，则选择
                if current_count < target_count and sample_scores.get(sample_idx, 0.5) > 0.6:
                    child.append(sample_idx)
                    used_classes[cls] = current_count + 1
            # 第二步：如果样本不足，补充其他高得分样本
            if len(child) < region_size:
                remaining = region_size - len(child)
                for sample_idx in sorted_samples:
                    if len(child) >= region_size:
                        break
                    if sample_idx not in child:
                        child.append(sample_idx)
            # 第三步：如果仍然不足，随机补充
            if len(child) < region_size:
                remaining = region_size - len(child)
                available = list(set(combined) - set(child))
                if len(available) >= remaining:
                    child.extend(random.sample(available, remaining))
                else:
                    child.extend(available)
                    # 如果还不够，允许重复
                    remaining_after = region_size - len(child)
                    if remaining_after > 0:
                        child.extend(random.choices(combined, k=remaining_after))
            return child[:region_size]
        # 不进行交叉时，选择相对排名得分较高的父代
        parent1_score = np.mean([self.relative_rank_scores.get(idx, 0.5) for idx in parent1]) if hasattr(self, 'relative_rank_scores') else 0.5
        parent2_score = np.mean([self.relative_rank_scores.get(idx, 0.5) for idx in parent2]) if hasattr(self, 'relative_rank_scores') else 0.5
        return parent1 if parent1_score > parent2_score else parent2
    def mutate(self, individual, data_size, y, region_size):
        """
        改进的变异操作：基于相对排名得分优化样本替换
        优先替换相对排名得分低的样本（向后排名比向前排名差的样本）
        """
        if random.random() < self.mutation_rate:
            mutated = individual.copy()
            if len(mutated) == 0:
                return mutated
            # 如果没有预计算数据，回退到随机变异
            if not hasattr(self, 'relative_rank_scores') or not self.relative_rank_scores:
                # 随机替换一个样本
                replace_index = random.randint(0, len(mutated) - 1)
                available_indices = list(set(range(data_size)) - set(mutated))
                if available_indices:
                    new_sample = random.choice(available_indices)
                    mutated[replace_index] = new_sample
                return mutated
            # 找到个体中相对排名得分最低的样本
            individual_scores = []
            for sample_idx in mutated:
                if sample_idx in self.relative_rank_scores:
                    score = self.relative_rank_scores[sample_idx]
                else:
                    # 动态计算
                    cls = y[sample_idx]
                    if hasattr(self, 'class_candidates') and cls in self.class_candidates:
                        cls_candidates = self.class_candidates[cls]
                        forward_rank = self._calculate_forward_rank(sample_idx, self.target_point, cls_candidates, self.X, cls)
                        backward_rank = self._calculate_backward_rank(sample_idx, self.target_point, cls_candidates, self.X, cls)
                        score = self._calculate_relative_rank_score(forward_rank, backward_rank)
                        self.relative_rank_scores[sample_idx] = score
                    else:
                        score = 0.5
                individual_scores.append((sample_idx, score))
            # 按得分升序排序，找到得分最低的样本
            individual_scores.sort(key=lambda x: x[1])
            worst_sample, worst_score = individual_scores[0]
            worst_class = y[worst_sample]
            # 从候选集中寻找相对排名得分更高的替代样本
            available_indices = list(self.candidate_set - set(mutated)) if hasattr(self, 'candidate_set') else list(set(range(data_size)) - set(mutated))
            if available_indices:
                # 计算候选样本的相对排名得分
                candidate_scores = {}
                for candidate_idx in available_indices:
                    if candidate_idx in self.relative_rank_scores:
                        score = self.relative_rank_scores[candidate_idx]
                    else:
                        # 动态计算
                        cls = y[candidate_idx]
                        if hasattr(self, 'class_candidates') and cls in self.class_candidates:
                            cls_candidates = self.class_candidates[cls]
                            forward_rank = self._calculate_forward_rank(candidate_idx, self.target_point, cls_candidates, self.X, cls)
                            backward_rank = self._calculate_backward_rank(candidate_idx, self.target_point, cls_candidates, self.X, cls)
                            score = self._calculate_relative_rank_score(forward_rank, backward_rank)
                            self.relative_rank_scores[candidate_idx] = score
                        else:
                            score = 0.5
                    candidate_scores[candidate_idx] = score
                # 优先选择同类别且得分更高的样本
                same_class_candidates = [idx for idx in available_indices 
                                       if y[idx] == worst_class and candidate_scores.get(idx, 0.5) > worst_score]
                if same_class_candidates:
                    # 选择同类别中得分最高的样本
                    best_candidate = max(same_class_candidates, key=lambda x: candidate_scores.get(x, 0.5))
                    replace_index = mutated.index(worst_sample)
                    mutated[replace_index] = best_candidate
                else:
                    # 选择其他类别中得分更高的样本
                    better_candidates = [idx for idx in available_indices 
                                       if candidate_scores.get(idx, 0.5) > worst_score]
                    if better_candidates:
                        best_candidate = max(better_candidates, key=lambda x: candidate_scores.get(x, 0.5))
                        replace_index = mutated.index(worst_sample)
                        mutated[replace_index] = best_candidate
                    else:
                        # 如果没有更好的样本，随机选择一个样本替换
                        new_sample = random.choice(available_indices)
                        replace_index = mutated.index(worst_sample)
                        mutated[replace_index] = new_sample
            return mutated
        return individual
    def optimize_region(self, X, y, target_point, target_class=None,region_size=20):
        """优化寻找最佳局部区域"""
        data_size = len(X)
        # 使用改进的初始化策略
        population = self.initialize_population(data_size, region_size, X, y, target_point)
        best_individual = None
        best_fitness = -1
        best_class_distribution = None
        print(f"开始遗传算法优化，寻找包含 {region_size} 个样本的局部区域...")
        print(f"目标类别: {target_class if target_class is not None else '未知'}")
        for generation in range(self.generations):
            fitness_scores = []
            class_distributions = []
            for ind in population:
                fitness = self.evaluate_fitness(ind, X, y, target_point, target_class)
                fitness_scores.append(fitness)
                # 记录类别分布
                y_region = y[ind]
                class_distributions.append(Counter(y_region))
            # 更新最佳个体
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()
                best_class_distribution = class_distributions[max_fitness_idx]
                # print(f"代 {generation}: 发现新最佳个体，适应度={best_fitness:.4f}, 类别分布={dict(best_class_distribution)}")
            # 选择精英个体
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            new_population = [population[i] for i in elite_indices]
            # 生成新一代
            while len(new_population) < self.population_size:
                parents = self.select_parents(population, fitness_scores)
                child = self.crossover(parents[0], parents[1],y,region_size)
                child = self.mutate(child, data_size, y, region_size)
                new_population.append(child)
            population = new_population
            if generation % 10 == 0:
                avg_fitness = np.mean(fitness_scores)
                # print(f"代 {generation}: 平均适应度={avg_fitness:.4f}, 最佳适应度={best_fitness:.4f}")
        print(f"遗传算法完成，最终最佳适应度: {best_fitness:.4f}")
        print("最佳个体",best_individual)
        return best_individual, best_fitness, best_class_distribution
# ==================== 多策略局部区域搜索 ====================
class MultiStrategyRegionSearcher:
    """多策略局部区域搜索器，提高找到正确类别的概率"""
    def __init__(self,inv_cov_matrix=None):
        self.ga_optimizer = ImprovedGeneticAlgorithmOptimizer(inv_cov_matrix=inv_cov_matrix)
        self.inv_cov_matrix = inv_cov_matrix
    def search_by_kmeans(self, X, y, target_point, region_size=20):
        """使用K-means聚类辅助搜索"""
        # 使用K-means将数据聚类
        n_clusters = min(10, len(X) // 10) # 适中的聚类数
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        # 找到目标点所属的簇
        target_cluster = kmeans.predict(target_point.reshape(1, -1))[0]
        # 提取该簇中的所有样本
        cluster_indices = np.where(clusters == target_cluster)[0]
        # 如果簇中的样本太多，选择距离最近的region_size个样本
        if len(cluster_indices) > region_size:
            cluster_points = X[cluster_indices]
            distances = np.linalg.norm(cluster_points - target_point, axis=1)
            nearest_indices = cluster_indices[np.argsort(distances)[:region_size]]
            return nearest_indices.tolist()
        else:
            return cluster_indices.tolist()
    def search_by_class_proximity(self, X, y, target_point, region_size=20):
        """基于类别邻近度的搜索"""
        # 计算所有样本与目标点的距离
        distances = np.linalg.norm(X - target_point, axis=1)
        # 对每个类别，选择距离最近的几个样本
        unique_classes = np.unique(y)
        samples_per_class = max(1, region_size // len(unique_classes))
        selected_indices = []
        for cls in unique_classes:
            class_indices = np.where(y == cls)[0]
            if len(class_indices) > 0:
                class_distances = distances[class_indices]
                # 选择该类别中距离最近的样本
                nearest_class_indices = class_indices[np.argsort(class_distances)[:samples_per_class]]
                selected_indices.extend(nearest_class_indices.tolist())
        # 如果样本数不足，用最近邻补充
        if len(selected_indices) < region_size:
            remaining = region_size - len(selected_indices)
            mask = np.ones(len(X), dtype=bool)
            mask[selected_indices] = False
            available_indices = np.where(mask)[0]
            if len(available_indices) > 0:
                available_distances = distances[available_indices]
                nearest_indices = available_indices[np.argsort(available_distances)[:remaining]]
                selected_indices.extend(nearest_indices.tolist())
        return selected_indices
    def multi_strategy_search(self, X, y, target_point, target_class=None, region_size=20):
        """多策略搜索，提高找到正确类别的概率"""
        print("开始多策略局部区域搜索...")
        # 策略1: 遗传算法搜索
        print("\n策略1: 遗传算法搜索")
        # print("*************搜索范围********************")
        # print(X)
        ga_region, ga_fitness, ga_class_dist = self.ga_optimizer.optimize_region(
            X, y, target_point, target_class, region_size
        )
        # 策略2: K-means聚类搜索
        # print("\n策略2: K-means聚类搜索")
        # kmeans_region = self.search_by_kmeans(X, y, target_point, region_size)
        # kmeans_class_dist = Counter(y[kmeans_region])
        # 策略3: 类别邻近度搜索
        # print("\n策略3: 类别邻近度搜索")
        # class_prox_region = self.search_by_class_proximity(X, y, target_point, region_size)
        # class_prox_class_dist = Counter(y[class_prox_region])
        # 评估各策略结果
        regions = [
            (ga_region, ga_fitness, ga_class_dist, "遗传算法"),
            # (kmeans_region, self.ga_optimizer.evaluate_fitness(kmeans_region, X, y, target_point, target_class), 
            #  kmeans_class_dist, "K-means"),
            # (class_prox_region, self.ga_optimizer.evaluate_fitness(class_prox_region, X, y, target_point, target_class),
            #  class_prox_class_dist, "类别邻近度")
        ]
        # 按适应度排序
        regions.sort(key=lambda x: x[1], reverse=True)
        # print("\n=== 多策略搜索结果比较 ===")
        for i, (region, fitness, class_dist, strategy) in enumerate(regions):
            target_ratio = class_dist.get(target_class, 0) / len(region) if target_class is not None else 0
            print(f"{i+1}. {strategy}: 适应度={fitness:.4f}, 目标类别比例={target_ratio:.4f}, 类别分布={dict(class_dist)}")
        # 返回最佳结果
        best_region, best_fitness, best_class_dist, best_strategy = regions[0]
        print(f"\n最佳策略: {best_strategy}")
        return best_region, best_fitness, best_class_dist
# ==================== 局部区域属性分析器 ====================
class LocalRegionAnalyzer:
    """分析局部区域内每个类别下每个属性的统计特性"""
    def __init__(self):
        pass
    def calculate_attribute_entropy(self, values, bins=100):
        """计算连续值数组的熵"""
        if len(values) < 2:
            return 0.0
        counts, _ = np.histogram(values, bins=bins)
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]
        entropy_val = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(bins)
        normalized_entropy = entropy_val / max_entropy if max_entropy > 0 else 0
        return normalized_entropy
    def calculate_attribute_variation(self, values):
        """计算属性的变异系数（离散度）"""
        if len(values) < 2:
            return 0.0
        mean_val = np.mean(values)
        std_val = np.std(values)
        if mean_val != 0:
            cv = std_val / abs(mean_val)
            stability = 1 / (1 + cv)
            return stability
        else:
            return 0.0
    def entropy_to_credibility(self, entropy_values):
        """将熵值转换为可信度"""
        entropy_values = np.array(entropy_values)
        credibility_scores = 1 - entropy_values
        return credibility_scores
    def calculate_mutual_info_per_class(self, X_region, y_region, feature_names):
        """在局部区域内计算每个属性对各个类别的信息增量"""
        n_features = X_region.shape[1]
        unique_classes = np.unique(y_region)
        mutual_info_results = {}
        for cls in unique_classes:
            y_binary = (y_region == cls).astype(int)
            if len(np.unique(y_binary)) > 1:
                mi_scores = mutual_info_classif(X_region, y_binary)
            else:
                mi_scores = np.zeros(n_features)
            mutual_info_results[cls] = mi_scores
        return mutual_info_results
    def analyze_local_region(self, X_region, y_region, feature_names):
        """综合分析局部区域内的属性特性"""
        unique_classes = np.unique(y_region)
        n_features = X_region.shape[1]
        # 存储结果
        entropy_results = {}
        credibility_results = {}
        variation_results = {}
        mutual_info_results = {}
        print(f"\n=== 局部区域属性分析 ===")
        print(f"局部区域样本数: {len(X_region)}")
        print(f"包含的类别: {unique_classes}")
        # 计算每个类别下每个属性的统计量
        for cls in unique_classes:
            # 提取当前类别的样本
            class_indices = (y_region == cls)
            X_class = X_region[class_indices]
            print(f"\n分析类别 {cls} (样本数: {np.sum(class_indices)})")
            class_entropies = []
            class_variations = []
            # 计算每个属性的熵和变异系数
            for feature_idx in range(n_features):
                feature_values = X_class[:, feature_idx]
                # 计算熵
                entropy_val = self.calculate_attribute_entropy(feature_values)
                class_entropies.append(entropy_val)
                # 计算变异系数（离散度）
                variation_val = self.calculate_attribute_variation(feature_values)
                class_variations.append(variation_val)
                # print(f" 属性 '{feature_names[feature_idx]}': 熵={entropy_val:.4f}, 离散度={variation_val:.4f}")
            entropy_results[cls] = class_entropies
            variation_results[cls] = class_variations
            # 计算可信度
            credibility_results[cls] = self.entropy_to_credibility(class_entropies)
        # 计算每个属性对各个类别的信息增量
        mutual_info_results = self.calculate_mutual_info_per_class(X_region, y_region, feature_names)
        # 打印信息增量结果
        # print(f"\n=== 局部区域内属性对各类别的信息增量 ===")
        # for cls, mi_scores in mutual_info_results.items():
        #     print(f"\n类别 {cls}:")
        #     for feature_idx, mi_score in enumerate(mi_scores):
                # print(f" 属性 '{feature_names[feature_idx]}': 信息增量={mi_score:.4f}")
        return {
            'entropy': entropy_results,
            'credibility': credibility_results,
            'variation': variation_results,
            'mutual_info': mutual_info_results
        }
    def calculate_global_mutual_info(self, X_region, y_region, feature_names):
        """计算全局互信息（所有类别）"""
        n_features = X_region.shape[1]
        if len(np.unique(y_region)) < 2:
            return np.zeros(n_features)
        # 计算全局互信息
        mi_scores = mutual_info_classif(X_region, y_region)
        # print(f"\n=== 全局信息增益（所有类别）===")
        # for feature_idx, mi_score in enumerate(mi_scores):
            # print(f"属性 '{feature_names[feature_idx]}': 全局信息增益={mi_score:.4f}")
        return mi_scores
    def calculate_comprehensive_weights(self, analysis_results, feature_names, alpha=0.4, beta=0.3, gamma=0.3):
        """计算综合权重：结合熵、可信度、离散度和信息增益"""
        unique_classes = list(analysis_results['entropy'].keys())
        n_features = len(feature_names)
        comprehensive_weights = {}
        # 计算全局信息增益的归一化权重
        global_mi = analysis_results.get('global_mutual_info', np.ones(n_features))
        global_mi_weights = global_mi / (np.sum(global_mi) + 1e-8)
        for cls in unique_classes:
            # 获取当前类别的各项指标
            credibilities = analysis_results['credibility'][cls]
            variations = analysis_results['variation'][cls]
            mutual_info = analysis_results['mutual_info'][cls]
            # 归一化各项指标
            cred_norm = credibilities / (np.sum(credibilities) + 1e-8)
            var_norm = variations / (np.sum(variations) + 1e-8)
            mi_norm = mutual_info / (np.sum(mutual_info) + 1e-8)
            # 计算综合权重
            weights = (alpha * cred_norm + # 可信度权重
                      beta * var_norm + # 离散度权重
                      gamma * mi_norm) # 信息增益权重
            # 进一步结合全局信息增益
            weights = 0.7 * weights + 0.3 * global_mi_weights
            comprehensive_weights[cls] = weights
        # 打印权重结果
        # print(f"\n=== 各类别属性综合权重 ===")
        # for cls, weights in comprehensive_weights.items():
            # print(f"\n类别 {cls}:")
            # for feature_idx, weight in enumerate(weights):
                # print(f" 属性 '{feature_names[feature_idx]}': 权重={weight:.4f}")
        return comprehensive_weights

# ==================== 加权KNN分类器 ====================
class WeightedKNNClassifier:
    """基于局部区域权重的加权KNN分类器"""
    def __init__(self, k=7, distance_weight_method='exponential', inv_cov_matrix=None):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.X_row_number = None
        self.feature_names = None
        self.analyzer = LocalRegionAnalyzer()
        self.distance_weight_method = distance_weight_method
        self.inv_cov_matrix = inv_cov_matrix
    def fit(self, X_train, y_train, X_row_number, feature_names=None):
        """训练模型"""
        self.X_train = X_train
        self.y_train = y_train
        self.n_classes = np.unique(y_train)
        self.X_row_number = X_row_number
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X_train.shape[1])]
        print(f"训练集大小: {X_train.shape}")
    def calculate_improved_distance_weight(self, distances, method='rank_exponential', top_m=5):
        """
        改进的距离权重计算，急剧放大最近邻的权重
        """
        if len(distances) == 0:
            return np.array([])
        distances = np.array(distances)
        if method == 'rank_exponential':
            # 基于排名的指数权重：最近邻权重最大，急剧下降
            ranks = np.arange(1, len(distances) + 1) # 排名：1, 2, 3, ...
            # 使用指数衰减，最近邻权重远大于其他
            base = 2.0 # 基数越大，衰减越快
            weights = base ** (-ranks + 1) # 排名1: 2^0=1, 排名2: 2^-1=0.5, 排名3: 2^-2=0.25
            weights = [ w*(1/(d+1e-8)) for w,d in zip(weights,distances)]
        elif method == 'rank_power':
            # 基于排名的幂次权重
            ranks = np.arange(1, len(distances) + 1)
            power = 3.0 # 幂次越大，衰减越快
            weights = 1.0 / (ranks ** power) 
            # weights = [ w*d for w,d in zip(weights,distances)]
        elif method == 'relative_distance':
            # 相对距离权重：基于与最近邻的相对距离
            min_distance = np.min(distances)
            if min_distance == 0:
                min_distance = 1e-8
            # 计算相对距离比率
            relative_distances = distances / min_distance
            # 使用指数衰减
            weights = np.exp(2.0 * (1 - relative_distances)) # 最近邻权重为e^2≈7.4，是第二近邻的e倍
        elif method == 'top_m_dominant':
            # 前m个主导邻居：只有前m个邻居有显著权重
            weights = np.zeros(len(distances))
            for i in range(min(top_m, len(distances))):
                if i == 0:
                    weights[i] = 1.0 # 最近邻权重最大
                else:
                    weights[i] = 0.1 # 其他前m个邻居权重很小
            # 其余邻居权重为0
        else:
            # 默认使用基于排名的指数权重
            ranks = np.arange(1, len(distances) + 1)
            weights = 2.0 ** (-ranks + 1)
        # 归一化权重
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        return weights
    def calculate_distance_weight(self, distances, method='exponential'):
        """计算距离权重，距离越小权重越大"""
        if len(distances) == 0:
            return np.array([])
        # 避免除零错误，添加一个小常数
        distances = np.array(distances) + 1e-8
        if method == 'inverse':
            # 使用距离的倒数作为权重
            weights = 1.0 / distances
        elif method == 'gaussian':
            # 使用高斯核函数作为权重
            sigma = np.median(distances) # 使用中位数作为带宽
            if sigma < 1e-8:
                sigma = 1.0
            weights = np.exp(-distances**2 / (2 * sigma**2))
        elif method == 'exponential':
            # 使用指数衰减作为权重
            weights = np.exp(-distances)
        else:
            # 默认使用距离的倒数
            weights = 1.0 / distances
        # 归一化权重
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        return weights
    def calculate_weighted_distance(self, target_point, weights):
        """计算加权欧式/马氏距离"""
        # 确保权重是归一化的
        weights = weights / np.sum(weights)
        # 计算加权距离
        weighted_distances = []
        for i in range(len(self.X_train)):
            # # 特征加权欧氏距离
            weighted_diff = weights * (self.X_train[i] - target_point)
            distance = np.sqrt(np.sum(weighted_diff ** 2))
            weighted_distances.append((self.X_row_number[i], distance, self.y_train[i]))
            # 特征加权马氏距离
            # P = np.diag(weights)
            # diff = self.X_train[i] - target_point
            # temp = P @ diff
            # weighted_distance = np.sqrt(temp.T @ self.inv_cov_matrix @ temp)
            # weighted_distances.append((i,weighted_distance, self.y_train[i]))
        return weighted_distances
    def predict_with_weights(self, target_point, global_weights, local_region_classes):
        """
        使用综合权重进行加权KNN预测
        comprehensive_weights: 每个类别的权重向量
        local_region_classes: 局部区域中包含的类别
        """
        print(f"\n=== 使用加权KNN进行预测 ===")
        # print(f"局部区域中包含的类别: {local_region_classes}")
        # 为每个类别计算加权距离并找到KNN
        knn_results = {}
        consistency_scores = {}
        
        # print(f"\n使用类别 {cls} 的权重计算距离...")
        # print(f"权重向量: {[f'{w:.4f}' for w in weights]}")
        # 计算加权距离
        weighted_distances = self.calculate_weighted_distance(target_point, global_weights)
        # 按距离排序，选择前k个邻居
        weighted_distances.sort(key=lambda x: x[1])
        k_neighbors = weighted_distances[:self.k]
        print("找到的k近邻为：",k_neighbors)
        for cls in self.n_classes:
            # 统计KNN中属于当前类别的样本数
            class_count = sum(1 for _, _, neighbor_class in k_neighbors if neighbor_class == cls)
            consistency = class_count / self.k
            knn_results[cls] = {
                'neighbors': k_neighbors,
                'class_count': class_count,
                'consistency': consistency
            }
            consistency_scores[cls] = consistency
        # print(f"类别 {cls}: KNN中属于该类别的样本数 = {class_count}/{self.k}, 一致性 = {consistency:.4f}")
        # print(f" 前5个最近邻: {[(idx, f'{dist:.4f}', cls) for idx, dist, cls in k_neighbors[:5]]}")
        # 选择一致性最高的类别作为预测结果
        if consistency_scores:
            predicted_class = max(consistency_scores.items(), key=lambda x: x[1])[0]
            max_consistency = consistency_scores[predicted_class]
            # print(f"\n预测结果: 类别 {predicted_class} (一致性: {max_consistency:.4f})")
            return predicted_class, max_consistency, knn_results
        else:
            print("警告: 没有找到有效的KNN结果")
            # 如果没有结果，使用简单的KNN
            distances = np.linalg.norm(self.X_train - target_point, axis=1)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_classes = self.y_train[nearest_indices]
            predicted_class = Counter(nearest_classes).most_common(1)[0][0]
            return predicted_class, 0.0, {}
    def predict_with_distance_weighted_voting(self, target_point, comprehensive_weights, local_region_classes):
        """
        使用距离加权投票进行KNN预测
        考虑邻居距离的权重，距离越近投票权重越大
        """
        print(f"\n=== 使用距离加权KNN进行预测 ===")
        # print(f"距离权重方法: {self.distance_weight_method}")
        # print(f"局部区域中包含的类别: {local_region_classes}")
        # print(f"局部区域中融合权重: {comprehensive_weights}")
        # print(comprehensive_weights)
        print("目标测试点",target_point)
        # 为每个类别计算加权距离并找到KNN
        knn_results = {}
        weighted_votes = {}
        simple_votes = {} # 简单投票（用于比较）
        for cls, weights in comprehensive_weights.items():
            if cls not in local_region_classes:
                print(f"跳过类别 {cls}，因为不在局部区域中")
                continue
            print(f"\n使用类别 {cls} 的权重计算距离...")
            # 计算加权距离
            weighted_distances = self.calculate_weighted_distance(target_point, weights)
            # 按距离排序，选择前k个邻居
            weighted_distances.sort(key=lambda x: x[1])
            k_neighbors = weighted_distances[:self.k]
            print(f"\n使用类别 {cls} 的权重得到的k个最近邻...")
            print(k_neighbors)
            # 提取距离和类别信息
            distances = [item[1] for item in k_neighbors]
            neighbor_classes = [item[2] for item in k_neighbors]
            # 计算距离权重
            distance_weights = self.calculate_improved_distance_weight(distances)
            class_count = sum(1 for _, _, neighbor_class in k_neighbors if neighbor_class == cls)
            consistency = class_count / self.k
            
            # 计算加权投票（考虑距离权重）
            weighted_vote = 0.0
            for rank, neighbor_class in enumerate(neighbor_classes):
                if neighbor_class == cls:
                    # weighted_vote += 1/(2**rank)
                    # weighted_vote += 1/(distances[rank]+1e-8)
                    weighted_vote = consistency* 1/(distances[rank]+1e-8)
                    break
            weighted_consistency = weighted_vote
            weighted_votes[cls] = weighted_consistency
            
            print(f"\n类别 {cls} 加权投票权重是：",weighted_consistency)
            # 存储详细结果
            knn_results[cls] = {
                'neighbors': k_neighbors,
                'distances': distances,
                'distance_weights': distance_weights,
                'neighbor_classes': neighbor_classes,
                'consistency': weighted_consistency,
                'weights_used': weights # 使用的特征权重
            }
            # print(f"类别 {cls}:")
            # print(f" 加权投票: {weighted_consistency:.4f}")
            # print(f" 距离权重: {[f'{w:.4f}' for w in distance_weights]}")
            # print(f" 邻居类别: {neighbor_classes}")
            # print(f" 邻居距离: {[f'{d:.4f}' for d in distances]}")
        # 使用加权投票选择预测结果
        if weighted_votes:
            predicted_class = max(weighted_votes.items(), key=lambda x: x[1])[0]
            max_weighted_consistency = weighted_votes[predicted_class]
            print(f"\n预测结果:")
            print(f" 加权投票预测: 类别 {predicted_class} (一致性: {max_weighted_consistency:.4f})")
            return predicted_class, max_weighted_consistency, knn_results, weighted_votes
        else:
            print("警告: 没有找到有效的KNN结果")
            # 使用简单的KNN作为备选
            distances = np.linalg.norm(self.X_train - target_point, axis=1)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_classes = self.y_train[nearest_indices]
            predicted_class = Counter(nearest_classes).most_common(1)[0][0]
            return predicted_class, 0.0, {}, {}, {}
    def evaluate_consistency(self, knn_results, true_class):
        """评估预测结果的一致性"""
        if not knn_results:
            return 0.0, {}
        # 计算每个类别的置信度
        confidence_scores = {}
        total_consistency = sum(result['consistency'] for result in knn_results.values())
        for cls, result in knn_results.items():
            if total_consistency > 0:
                confidence = result['consistency'] / total_consistency
            else:
                confidence = 0.0
            confidence_scores[cls] = confidence
        # 真实类别的一致性（如果真实类别在结果中）
        true_class_consistency = knn_results.get(true_class, {}).get('consistency', 0.0)
        true_class_confidence = confidence_scores.get(true_class, 0.0)
        print(f"\n=== 一致性评估 ===")
        print(f"真实类别: {true_class}")
        print(f"真实类别一致性: {true_class_consistency:.4f}")
        print(f"真实类别置信度: {true_class_confidence:.4f}")
        for cls, confidence in confidence_scores.items():
            print(f"类别 {cls} 置信度: {confidence:.4f}")
        return true_class_consistency, confidence_scores
# ==================== 局部区域属性分析器（质心法） ====================
class CorrectedLocalRegionAnalyzer:
    """修正的局部区域分析器 - 直接使用原始比例计算，无需额外标准化"""
    def __init__(self, normalize_weights=False, epsilon=1e-8):
        self.normalize_weights = normalize_weights
        self.epsilon = epsilon # 防止除零的小常数
    def _calculate_safe_ratios(self, values, centroid):
        """
        安全的比例计算，处理质心接近0的情况
        使用相对差异而不是绝对比例，避免尺度问题
        """
        ratios = np.zeros_like(values)
        for i in range(values.shape[1]): # 对每个特征
            centroid_val = centroid[i]
            feature_values = values[:, i]
            if abs(centroid_val) > self.epsilon:
                # 正常情况：使用比例
                ratios[:, i] = feature_values / centroid_val
            else:
                # 质心接近0的特殊处理
                if np.all(np.abs(feature_values) < self.epsilon):
                    # 如果所有值都接近0，比例设为1（表示一致）
                    ratios[:, i] = 1.0
                else:
                    # 使用相对差异： (value - centroid) / (|centroid| + |value| + epsilon)
                    # 这能处理质心为0的情况，同时保持合理的数值范围
                    denominator = np.abs(centroid_val) + np.abs(feature_values) + self.epsilon
                    ratios[:, i] = (feature_values - centroid_val) / denominator
        return ratios
    def analyze_local_region(self, X_region, y_region, feature_names):
        """
        分析局部区域 - 直接使用比例计算，保持原始尺度信息
        """
        n_samples, n_features = X_region.shape
        unique_classes = np.unique(y_region)
        print(f"分析局部区域: {n_samples}个样本, {n_features}个特征")
        # print(f"数据尺度范围 - 最小值: {np.min(X_region):.4f}, 最大值: {np.max(X_region):.4f}")
        # 1. 计算质心（保持原始尺度）
        centroid = np.mean(X_region, axis=0)
        # print(f"质心计算完成 - 尺度范围: {np.min(centroid):.4f} 到 {np.max(centroid):.4f}")
        print(f"质心计算完成 - {centroid}")
        comprehensive_weights = {}
        region_info = {
            'centroid': centroid,
            'class_stats': {},
            'feature_ranges': {feature: (np.min(X_region[:, i]), np.max(X_region[:, i])) 
                             for i, feature in enumerate(feature_names)}
        }
        for cls in unique_classes:
            print(f"\n分析类别 {cls}:")
            # 获取当前类别的所有点
            class_mask = (y_region == cls)
            X_class = X_region[class_mask]
            n_class_samples = len(X_class)
            if n_class_samples == 0:
                # 如果没有该类别的点，使用均匀权重
                weights = np.ones(n_features) / n_features
                comprehensive_weights[cls] = dict(zip(feature_names, weights))
                region_info['class_stats'][cls] = {
                    'n_samples': 0, 
                    'method': 'uniform (no samples)',
                    'weight_calculation': 'equal weights'
                }
                continue
            print(f" 类别 {cls} 有 {n_class_samples} 个样本")
            # 2. 计算安全的比例
            ratios = self._calculate_safe_ratios(X_class, centroid)
            # 3. 分析比例分布
            mean_ratios = np.mean(ratios, axis=0)
            std_ratios = np.std(ratios, axis=0)
            # 4. 使用几何平均（对比例数据更合适）
            # 几何平均能更好地处理比例数据的乘法性质
            valid_ratios = np.clip(ratios, 1e-8, 1e8) # 避免log(0)和数值溢出
            geometric_mean = np.exp(np.mean(np.log(valid_ratios), axis=0))
            # 5. 解释比例的含义
            # 比例 > 1: 该类别的点在该特征上倾向于高于区域平均水平
            # 比例 < 1: 该类别的点在该特征上倾向于低于区域平均水平
            # 比例 ≈ 1: 该类别的点在该特征上与区域平均水平相似
            # 6. 计算权重 - 使用比例偏离1的程度作为重要性指标
            # 偏离越大（无论是大于1还是小于1），说明该特征对区分该类更重要
            deviation_from_one = np.abs(geometric_mean - 1.0)
            # 7. 可选归一化
            if self.normalize_weights == False:
                if np.sum(mean_ratios) > 0:
                    weights = mean_ratios / np.sum(mean_ratios)
                else:
                    weights = np.ones(n_features) / n_features
            else:
                weights = mean_ratios
            comprehensive_weights[cls] = weights
            # 存储详细统计信息
            region_info['class_stats'][cls] = {
                'n_samples': n_class_samples,
                'mean_ratios': mean_ratios,
                'std_ratios': std_ratios,
                'geometric_mean_ratios': geometric_mean,
                'deviation_from_one': deviation_from_one,
                'final_weights': weights,
                'weight_calculation': 'deviation from centroid ratio'
            }
            print(f" 计算平均",region_info['class_stats'][cls]['mean_ratios'])
            print(f" 几何平均",region_info['class_stats'][cls]['geometric_mean_ratios'])
            print(f" final_weights",region_info['class_stats'][cls]['final_weights'])
            # 显示权重信息
            # top_features = sorted(zip(feature_names, weights, geometric_mean), 
            #                      key=lambda x: x[1], reverse=True)[:5]
            # print(f" 最重要的属性 (基于与质心的偏离程度):")
            # for feature, weight, ratio in top_features:
            #     deviation_type = "高于" if ratio > 1.0 else "低于"
            #     deviation_pct = abs(ratio - 1.0) * 100
            #     print(f" {feature}: 权重={weight:.4f}, 比例={ratio:.3f} ({deviation_type}质心{deviation_pct:.1f}%)")
        return {
            'comprehensive_weights': comprehensive_weights,
            'region_info': region_info
        }
# ==================== 确保局部区域包含所有类别的代表性样本 ====================
def ensure_class_diversity_in_region(X_train, y_train, target_point, inv_cov=None, region_size=15, 
                                   global_class_dist=None, max_samples_per_class=None):
    """
    确保局部区域包含所有类别的代表性样本
    参数:
        X_train: 训练集特征
        y_train: 训练集标签
        target_point: 目标点（测试样本）
        region_size: 区域大小
        global_class_dist: 全局类别分布（如果不提供，则从y_train计算）
        max_samples_per_class: 每个类别最大样本数（默认根据比例计算）
        inv_cov: 如果使用马氏距离
    返回:
        diverse_region: 包含多样性的局部区域索引
        region_class_dist: 区域内的类别分布
    """
    def mahalanobis_distance(point_set, target , inv_cov):
        distances=[]
        for point in point_set:
            diff = point- target
            temp = np.dot(diff,inv_cov)
            dis = np.sqrt(np.dot(temp,diff))
            distances.append(dis)
        return distances
    def attribute_alignment_similarity(point_set, target):
        """属性对齐相似性 - 评估每个属性上的相对关系"""
        n_features = len(target)
        diffences = []
        # 计算每个属性上的相对差异
        for point in point_set:
            
            attribute_differences = []
            for i in range(n_features):
                if point[i] == 0 and target[i] == 0:
                    # 两个点在该属性上都是0，认为完全相似
                    attr_diff = 0
                else:
                    max_val = max(abs(point[i]), abs(target[i]))
                    attr_diff = min(abs(point[i] - target[i]) / max_val, 1.0)
                
                attribute_differences.append(attr_diff)
                total_diff = sum(attribute_differences)
            diffences.append(total_diff)
        # 计算几何平均（对比例数据更合适）
        # geometric_mean = np.exp(np.mean(np.log(np.array(attribute_similarities) + 1e-8)))
        return diffences
    # 计算全局类别分布
    if global_class_dist is None:
        global_class_dist = Counter(y_train)
    unique_classes = np.unique(y_train)
    n_classes = len(unique_classes)
    print(f"构建多样性局部区域: 目标大小={region_size}, 类别数={n_classes}")
    print(f"全局类别分布: {dict(global_class_dist)}")
    # 计算每个类别在区域中的目标样本数
    if max_samples_per_class is None:
        # 根据全局比例分配样本数，但确保每个类别至少有1个样本
        total_samples = sum(global_class_dist.values())
        class_proportions = {cls: count/total_samples for cls, count in global_class_dist.items()}
        # 分配样本数，确保每个类别至少有1个样本
        class_targets = {}
        remaining_size = region_size
        # 按比例分配剩余样本
        for cls in unique_classes:
            if remaining_size <= 0:
                break
            additional = max(1, int(class_proportions[cls] * remaining_size))
            class_targets[cls] = min(additional, remaining_size)
            remaining_size -= additional
        # 如果还有剩余，按比例分配
        while remaining_size > 0:
            for cls in unique_classes:
                if remaining_size <= 0:
                    break
                class_targets[cls] += 1
                remaining_size -= 1
    else:
        # 使用固定的最大样本数
        class_targets = {cls: min(max_samples_per_class, global_class_dist[cls]) 
                        for cls in unique_classes}
    print(f"目标类别分布: {class_targets}")
    # 为每个类别选择最近的样本
    diverse_region = []
    for cls in unique_classes:
        # 获取该类别的所有样本
        class_indices = np.where(y_train == cls)[0]
        if len(class_indices) == 0:
            print(f"警告: 类别 {cls} 在训练集中没有样本")
            continue
        target_count = class_targets[cls]
        actual_count = min(target_count, len(class_indices))
        if actual_count == 0:
            continue
        # 计算该类样本与目标点的距离
        class_points = X_train[class_indices]
        # distances = np.linalg.norm(class_points - target_point, axis=1)
        
        distances = mahalanobis_distance(class_points,target_point,inv_cov)
        # 选择距离最近的样本
        # differences = attribute_alignment_similarity(class_points,target_point)
        # nearest_indices = class_indices[np.argsort(differences)[:actual_count]]
        nearest_indices = class_indices[np.argsort(distances)[:actual_count]]
        diverse_region.extend(nearest_indices.tolist())
        print(f" 类别 {cls}: 目标{target_count}个, 实际{len(nearest_indices)}个")
    # 如果样本数不足，用最近邻补充
    if len(diverse_region) < region_size:
        remaining = region_size - len(diverse_region)
        print(f"样本不足，补充{remaining}个最近邻")
        # 计算所有样本与目标点的距离
        # all_distances = np.linalg.norm(X_train - target_point, axis=1)
        all_differences = attribute_alignment_similarity(X_train,target_point)
        # 排除已选择的样本
        mask = np.ones(len(X_train), dtype=bool)
        mask[diverse_region] = False
        available_indices = np.where(mask)[0]
        if len(available_indices) > 0:
            # 选择距离最近的样本补充
            available_distances = all_differences[available_indices]
            nearest_indices = available_indices[np.argsort(available_distances)[:remaining]]
            diverse_region.extend(nearest_indices.tolist())
    # 如果样本数超过，随机删除
    if len(diverse_region) > region_size:
        diverse_region = random.sample(diverse_region, region_size)
    region_class_dist = Counter(y_train[diverse_region])
    print(f"构建完成: 区域大小={len(diverse_region)}, 类别分布={dict(region_class_dist)}")
    return diverse_region, region_class_dist
class BinaryClassWeightCorrector(BaseEstimator):
    """
    基于二分类全局流形约束的权重向量矫正器
    针对每个类别c，将其视为二分类问题（c vs. 非c），
    然后通过广义瑞利商流形约束矫正局部权重向量。
    Parameters
    ----------
    reg_param : float, default=1e-6
        正则化参数，防止数值不稳定
    alpha : float, default=0.5
        权衡参数，平衡局部权重保持和全局流形约束
    method : str, default='projection'
        矫正方法，可选 'projection' 或 'optimization'
    Attributes
    ----------
    X_train_ : array-like
        训练数据特征
    y_train_ : array-like
        训练数据标签
    n_features_ : int
        特征数量
    class_stats_ : dict
        每个类别的统计信息缓存
    """
    def __init__(self, reg_param=1e-6, alpha=0.5, method='projection'):
        self.reg_param = reg_param
        self.alpha = alpha
        self.method = method
        self.class_stats_ = {}
    def fit(self, X_train, y_train):
        """
        拟合训练数据，计算每个类别的统计信息
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            训练数据特征
        y_train : array-like of shape (n_samples,)
            训练数据标签
        Returns
        -------
        self : object
            返回拟合后的模型实例
        """
        X_train, y_train = check_X_y(X_train, y_train)
        self.X_train_ = X_train
        self.y_train_ = y_train
        self.n_features_ = X_train.shape[1]
        self.classes_ = np.unique(y_train)
        # 计算每个类别的统计信息
        self._compute_class_statistics()
        return self
    def _compute_class_statistics(self):
        """计算每个类别的统计信息"""
        for cls in self.classes_:
            # 获取类别c的数据
            X_c = self.X_train_[self.y_train_ == cls]
            X_not_c = self.X_train_[self.y_train_ != cls]
            n_c = X_c.shape[0]
            n_not_c = X_not_c.shape[0]
            # 计算均值
            mean_c = np.mean(X_c, axis=0)
            mean_not_c = np.mean(X_not_c, axis=0)
            global_mean = (n_c * mean_c + n_not_c * mean_not_c) / (n_c + n_not_c)
            # 计算类内散度矩阵
            S_w_c = self._compute_within_class_scatter(X_c, mean_c)
            S_w_not_c = self._compute_within_class_scatter(X_not_c, mean_not_c)
            S_w = S_w_c + S_w_not_c
            # 计算类间散度矩阵
            S_b = self._compute_between_class_scatter_binary(mean_c, mean_not_c, n_c, n_not_c, global_mean)
            # 存储统计信息
            self.class_stats_[cls] = {
                'S_w': S_w,
                'S_b': S_b,
                'mean_c': mean_c,
                'mean_not_c': mean_not_c,
                'global_mean': global_mean,
                'n_c': n_c,
                'n_not_c': n_not_c
            }
    def _compute_within_class_scatter(self, X, mean):
        """计算类内散度矩阵"""
        n_samples, n_features = X.shape
        S_w = np.zeros((n_features, n_features))
        for i in range(n_samples):
            diff = (X[i] - mean).reshape(-1, 1)
            S_w += diff @ diff.T
        return S_w
    def _compute_between_class_scatter_binary(self, mean_c, mean_not_c, n_c, n_not_c, global_mean):
        """计算二分类的类间散度矩阵"""
        # 方法1: 使用标准的LDA公式
        diff_c = (mean_c - global_mean).reshape(-1, 1)
        diff_not_c = (mean_not_c - global_mean).reshape(-1, 1)
        S_b = n_c * (diff_c @ diff_c.T) + n_not_c * (diff_not_c @ diff_not_c.T)
        # 方法2: 简化的二分类公式 (等价形式)
        # diff = (mean_c - mean_not_c).reshape(-1, 1)
        # S_b_simple = (n_c * n_not_c / (n_c + n_not_c)) * (diff @ diff.T)
        return S_b
    def correct_weight_vector(self, w_local, class_label):
        """
        矫正局部权重向量
        Parameters
        ----------
        w_local : array-like of shape (n_features,)
            待矫正的局部权重向量
        class_label : int
            类别标签，用于确定使用哪个二分类问题
        Returns
        -------
        w_corrected : array-like of shape (n_features,)
            矫正后的权重向量
        """
        if not hasattr(self, 'X_train_'):
            raise ValueError("模型未拟合，请先调用fit方法")
        if class_label not in self.class_stats_:
            raise ValueError(f"类别 {class_label} 不在训练数据中")
        w_local = np.array(w_local).flatten()
        if len(w_local) != self.n_features_:
            raise ValueError(f"权重向量维度不匹配: 期望{self.n_features_}，得到{len(w_local)}")
        # 获取该类别的统计信息
        stats = self.class_stats_[class_label]
        S_w = stats['S_w']
        S_b = stats['S_b']
        # 添加正则化
        S_w_reg = S_w + self.reg_param * np.eye(self.n_features_)
        if self.method == 'projection':
            w_corrected = self._project_to_manifold(w_local, S_b, S_w_reg)
        elif self.method == 'optimization':
            w_corrected = self._optimize_with_constraints(w_local, S_b, S_w_reg)
        else:
            raise ValueError(f"未知的矫正方法: {self.method}")
        # 归一化
        w_corrected = self._normalize_weight_vector(w_corrected)
        return w_corrected
    def _project_to_manifold(self, w_local, S_b, S_w):
        """投影到广义瑞利商流形"""
        # 求解广义特征值问题: S_b * w = λ * S_w * w
        try:
            # 使用scipy的eigh求解广义特征值问题
            from scipy.linalg import eigh
            eigenvalues, eigenvectors = eigh(S_b, S_w)
            # 按特征值降序排列
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            # 选择最大特征值对应的特征向量
            w_manifold = eigenvectors[:, 0]
            # 计算投影系数
            coeff = np.dot(w_manifold, w_local) / np.dot(w_manifold, w_manifold)
            # 投影到流形方向
            w_projected = coeff * w_manifold
            # 与原始向量结合
            w_corrected = self.alpha * w_projected + (1 - self.alpha) * w_local
            return w_corrected
        except np.linalg.LinAlgError:
            # 如果求解失败，返回原始向量
            return w_local
    def _optimize_with_constraints(self, w_local, S_b, S_w):
        """通过优化方法矫正权重向量"""
        def objective_function(w, w_local, S_b, S_w, alpha):
            """目标函数：平衡局部权重保持和广义瑞利商最大化"""
            w = w.reshape(-1, 1)
            w_local = w_local.reshape(-1, 1)
            # 项1: 与局部权重的差异
            diff_term = np.sum((w - w_local)**2)
            # 项2: 广义瑞利商 (最大化)
            denominator = w.T @ S_w @ w
            if denominator > 1e-10:
                rayleigh_quotient = (w.T @ S_b @ w) / denominator
            else:
                rayleigh_quotient = 0
            # 我们希望最大化瑞利商，所以在目标函数中取负
            return alpha * diff_term - (1 - alpha) * rayleigh_quotient
        # 使用L-BFGS-B优化
        result = minimize(
            objective_function,
            w_local,
            args=(w_local, S_b, S_w, self.alpha),
            method='L-BFGS-B',
            bounds=[(-1, 1)] * len(w_local), # 限制权重在[-1, 1]范围内
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        if result.success:
            return result.x
        else:
            # 如果优化失败，返回原始向量
            return w_local
    def _normalize_weight_vector(self, w):
        """归一化权重向量"""
        w_norm = np.linalg.norm(w)
        if w_norm > 1e-10:
            return w / w_norm
        else:
            return w
    def compute_binary_rayleigh_quotient(self, w, class_label):
        """计算权重向量在指定类别二分类问题上的广义瑞利商"""
        if class_label not in self.class_stats_:
            raise ValueError(f"类别 {class_label} 不在训练数据中")
        stats = self.class_stats_[class_label]
        S_w = stats['S_w'] + self.reg_param * np.eye(self.n_features_)
        S_b = stats['S_b']
        w = w.reshape(-1, 1)
        numerator = w.T @ S_b @ w
        denominator = w.T @ S_w @ w
        if denominator > 1e-10:
            return (numerator / denominator).flatten()[0]
        else:
            return 0.0
    def correct_weight_matrix(self, W_local, class_labels=None):
        """
        矫正整个权重矩阵
        Parameters
        ----------
        W_local : array-like of shape (n_classes, n_features)
            局部权重矩阵
        class_labels : array-like, optional
            类别标签列表，如果为None则使用训练数据中的所有类别
        Returns
        -------
        W_corrected : array-like of shape (n_classes, n_features)
            矫正后的权重矩阵
        """
        if class_labels is None:
            class_labels = self.classes_
        n_classes = len(class_labels)
        n_features = W_local.shape[1] if len(W_local.shape) > 1 else W_local.shape[0]
        W_corrected = np.zeros((n_classes, n_features))
        for i, cls in enumerate(class_labels):
            w_local = W_local[i] if len(W_local.shape) > 1 else W_local
            w_corrected = self.correct_weight_vector(w_local, cls)
            W_corrected[i] = w_corrected
        return W_corrected
    def get_correction_info(self, w_local, w_corrected, class_label):
        """获取矫正过程的详细信息"""
        info = {
            'rayleigh_quotient_local': self.compute_binary_rayleigh_quotient(w_local, class_label),
            'rayleigh_quotient_corrected': self.compute_binary_rayleigh_quotient(w_corrected, class_label),
            'cosine_similarity': np.dot(w_local, w_corrected) / 
                               (np.linalg.norm(w_local) * np.linalg.norm(w_corrected) + 1e-10),
            'norm_change': np.linalg.norm(w_corrected - w_local),
            'class_label': class_label
        }
        return info
class ManifoldWeightCorrector:
    """
    流形权重矫正器
    利用协方差矩阵的黎曼流形几何来矫正传统权重向量
    """
    def __init__(self, method='log_euclidean', alpha=0.5, reg_param=1e-6):
        """
        初始化参数
        Parameters:
        method: 流形距离计算方法 ('log_euclidean', 'affine_invariant')
        alpha: 流形矫正强度 (0-1之间)
        reg_param: 正则化参数，确保矩阵可逆
        """
        self.method = method
        self.alpha = alpha
        self.reg_param = reg_param
        self.global_cov = None
        self.class_manifold_centers = {}
        self.class_covariances = {}
    def fit(self, X_region, y_region):
        """
        在区域数据上拟合流形模型
        Parameters:
        X_region: 区域数据矩阵 (n_samples, n_features)
        y_region: 对应的标签向量
        """
        self.X_region = np.array(X_region)
        self.y_region = np.array(y_region)
        self.n_features = X_region.shape[1]
        # 计算全局协方差矩阵（正则化处理）
        self.global_cov = self._compute_regularized_covariance(X_region)
        # 按类别处理
        unique_classes = np.unique(y_region)
        for class_label in unique_classes:
            # 获取当前类别的样本
            class_mask = (y_region == class_label)
            X_class = X_region[class_mask]
            if len(X_class) < 2:
                # 样本太少，使用全局协方差
                self.class_covariances[class_label] = self.global_cov
                self.class_manifold_centers[class_label] = self.global_cov
                continue
            # 计算类别协方差矩阵
            class_cov = self._compute_regularized_covariance(X_class)
            self.class_covariances[class_label] = class_cov
            # 计算类别的流形中心（这里简化为类别协方差矩阵）
            # 在实际应用中可以使用Fréchet均值
            self.class_manifold_centers[class_label] = class_cov
        return self
    def correct_weights(self, original_weights, class_label, X_class=None):
        """
        矫正权重向量，融入流形几何信息
        Parameters:
        original_weights: 原始权重向量 (如广义瑞利商计算的权重)
        class_label: 当前类别标签
        X_class: 可选的类别数据（用于更精细的矫正）
        """
        if class_label not in self.class_manifold_centers:
            return original_weights
        # 获取类别流形中心
        class_manifold_center = self.class_manifold_centers[class_label]
        # 计算流形感知的特征重要性
        manifold_importance = self._compute_manifold_importance(
            class_manifold_center, self.global_cov)
        # 融合原始权重和流形重要性
        corrected_weights = self._fuse_weights(
            original_weights, manifold_importance)
        return corrected_weights
    def _compute_regularized_covariance(self, X):
        """计算正则化的协方差矩阵"""
        try:
            # 使用Ledoit-Wolf收缩估计器提高数值稳定性
            cov_estimator = LedoitWolf().fit(X)
            covariance = cov_estimator.covariance_
            # 添加小的正则化项确保正定性
            n_features = covariance.shape[0]
            covariance += self.reg_param * np.eye(n_features)
            return covariance
        except Exception as e:
            print(f"协方差计算错误: {e}，使用单位矩阵")
            n_features = X.shape[1]
            return np.eye(n_features)
    def _compute_manifold_importance(self, class_cov, global_cov):
        """
        基于流形几何计算特征重要性
        核心思想：在流形上距离全局结构越远的方向，可能越重要
        """
        try:
            if self.method == 'log_euclidean':
                return self._log_euclidean_importance(class_cov, global_cov)
            elif self.method == 'affine_invariant':
                return self._affine_invariant_importance(class_cov, global_cov)
            else:
                return self._default_importance(class_cov, global_cov)
        except Exception as e:
            print(f"流形重要性计算错误: {e}")
            return np.ones(self.n_features) / self.n_features
    def _log_euclidean_importance(self, class_cov, global_cov):
        """对数欧氏流形重要性计算"""
        # 计算矩阵对数差异
        try:
            log_class = logm(class_cov)
            log_global = logm(global_cov)
            log_diff = log_class - log_global
            # 特征重要性基于对数差异的行/列范数
            feature_importance = np.zeros(self.n_features)
            for i in range(self.n_features):
                # 计算每个特征在流形切空间中的"变化幅度"
                row_importance = np.linalg.norm(log_diff[i, :])
                col_importance = np.linalg.norm(log_diff[:, i])
                feature_importance[i] = (row_importance + col_importance) / 2
            # 归一化
            if np.sum(feature_importance) > 0:
                feature_importance = feature_importance / np.sum(feature_importance)
            else:
                feature_importance = np.ones(self.n_features) / self.n_features
            return feature_importance
        except Exception as e:
            print(f"对数欧氏计算错误: {e}")
            return np.ones(self.n_features) / self.n_features
    def _affine_invariant_importance(self, class_cov, global_cov):
        """仿射不变流形重要性计算"""
        try:
            # 计算仿射不变距离相关的特征重要性
            # 通过特征值分解分析流形结构
            eigenvals, eigenvecs = np.linalg.eigh(class_cov)
            global_eigenvals, global_eigenvecs = np.linalg.eigh(global_cov)
            # 特征值比率表示局部与全局结构的差异
            epsilon = 1e-8
            eigen_ratio = np.log((eigenvals + epsilon) / (global_eigenvals + epsilon))
            eigen_ratio = np.abs(eigen_ratio)
            # 将特征值差异映射到原始特征空间
            feature_importance = np.zeros(self.n_features)
            for i in range(self.n_features):
                for j in range(self.n_features):
                    # 特征i在特征向量j上的投影权重
                    projection_weight = eigenvecs[i, j] ** 2
                    feature_importance[i] += projection_weight * eigen_ratio[j]
            # 归一化
            if np.sum(feature_importance) > 0:
                feature_importance = feature_importance / np.sum(feature_importance)
            else:
                feature_importance = np.ones(self.n_features) / self.n_features
            return feature_importance
        except Exception as e:
            print(f"仿射不变计算错误: {e}")
            return np.ones(self.n_features) / self.n_features
    def _default_importance(self, class_cov, global_cov):
        """默认重要性计算：基于协方差矩阵的差异"""
        try:
            # 简单的基于Frobenius范数的重要性
            cov_diff = class_cov - global_cov
            feature_importance = np.zeros(self.n_features)
            for i in range(self.n_features):
                feature_importance[i] = np.linalg.norm(cov_diff[i, :])
            if np.sum(feature_importance) > 0:
                feature_importance = feature_importance / np.sum(feature_importance)
            else:
                feature_importance = np.ones(self.n_features) / self.n_features
            return feature_importance
        except Exception as e:
            print(f"默认重要性计算错误: {e}")
            return np.ones(self.n_features) / self.n_features
    def _fuse_weights(self, original_weights, manifold_importance):
        """融合原始权重和流形重要性"""
        # 使用alpha参数控制融合强度
        fused_weights = (1 - self.alpha) * original_weights + \
                       self.alpha * manifold_importance
        # 确保权重为正并归一化
        fused_weights = np.maximum(fused_weights, 0)
        if np.sum(fused_weights) > 0:
            fused_weights = fused_weights / np.sum(fused_weights)
        else:
            fused_weights = np.ones_like(fused_weights) / len(fused_weights)
        return fused_weights
    def compute_manifold_distance(self, cov1, cov2):
        """计算两个协方差矩阵在流形上的距离"""
        try:
            if self.method == 'log_euclidean':
                # 对数欧氏距离
                log_cov1 = logm(cov1)
                log_cov2 = logm(cov2)
                distance = np.linalg.norm(log_cov1 - log_cov2, 'fro')
            elif self.method == 'affine_invariant':
                # 仿射不变距离近似
                inv_sqrt_cov1 = fractional_matrix_power(cov1, -0.5)
                middle_matrix = inv_sqrt_cov1 @ cov2 @ inv_sqrt_cov1
                eigenvals = np.linalg.eigvalsh(middle_matrix)
                distance = np.sqrt(np.sum(np.log(eigenvals) ** 2))
            else:
                # 默认使用Frobenius距离
                distance = np.linalg.norm(cov1 - cov2, 'fro')
            return distance
        except Exception as e:
            print(f"流形距离计算错误: {e}")
            return np.linalg.norm(cov1 - cov2, 'fro')
class GlobalWeightMatrix:
    def __init__(self, method='lda'):
        self.method = method
        self.global_weights = None
        self.class_weight_dict = None # 新增：类别权重字典
        self.feature_names = None
        self.class_names = None
        self.scaler = StandardScaler()
    def fit(self, X, y=None, feature_names=None, class_names=None):
        X = np.array(X)
        if y is not None:
            y = np.array(y)
        self.feature_names = feature_names
        self.class_names = class_names
        X_scaled = self.scaler.fit_transform(X)
        if y is None:
            self.global_weights = self._compute_global_weights(X_scaled)
            self.global_weights = self.global_weights.reshape(1, -1)
            # 无监督情况下的字典
            self.class_weight_dict = {'global': self.global_weights[0]}
        else:
            unique_classes = np.unique(y)
            n_classes = len(unique_classes)
            n_features = X.shape[1]
            self.global_weights = np.zeros((n_classes, n_features))
            self.class_weight_dict = {} # 初始化字典
            for i, class_label in enumerate(unique_classes):
                weight_vector = self._compute_global_weights(X_scaled, y, class_label)
                self.global_weights[i] = weight_vector
                # 同时保存原始标签作为键，确保两种方式都能访问
                self.class_weight_dict[class_label] = weight_vector
            print(f"权重字典构建完成，包含 {len(self.class_weight_dict)} 个类别")
        return self

    def get_weight_matrix(self):
        """返回权重矩阵"""
        return self.global_weights
    def get_class_weight_dict(self):
        """返回类别权重字典"""
        return self.class_weight_dict
    def get_weight_by_class(self, class_identifier):
        """
        通过类别标识符获取权重向量
        支持类别名或原始标签
        """
        if self.class_weight_dict is None:
            raise ValueError("请先调用fit方法训练模型")
        if class_identifier in self.class_weight_dict:
            return self.class_weight_dict[class_identifier]
        else:
            # 尝试将输入转换为字符串格式
            str_identifier = str(class_identifier)
            if str_identifier in self.class_weight_dict:
                return self.class_weight_dict[str_identifier]
            else:
                print(f"警告: 未找到类别 '{class_identifier}' 的权重，使用均匀权重")
                n_features = self.global_weights.shape[1] if self.global_weights is not None else 1
                return np.ones(n_features) / n_features
    def _compute_global_weights(self, X, y=None, current_class=None):
        if y is not None and current_class is not None:
            if self.method == 'inter_class_difference':
                return self._inter_class_difference_weights(X, y, current_class)
            elif self.method == 'f_score':
                return self._f_score_weights(X, y, current_class)
            elif self.method == 'centroid':
                return self._centroid_weights(X, y, current_class)
            elif self.method == 'lda':
                return self._rayleigh_quotient(X, y, current_class)
            else:
                return self._inverse_covariance_weights(X, y, current_class)
        else:
            return self._inverse_covariance_weights(X, y, current_class)
    def _rayleigh_quotient(self, X, y, current_class):
        """
        基于广义瑞利商的LDA权重计算方法 - 二分类模式（当前类别 vs 其他类别）
        """
        try:
            # 创建二分类标签：当前类别 vs 其他类别
            y_binary = (y == current_class).astype(int)
            # 获取两类样本
            class1_mask = (y_binary == 1) # 当前类别
            class0_mask = (y_binary == 0) # 其他类别
            X1 = X[class1_mask] # 当前类别样本
            X0 = X[class0_mask] # 其他类别样本
            n1 = len(X1)
            n0 = len(X0)
            n_features = X.shape[1]
            # 检查样本数量是否足够
            if n1 < 2 or n0 < 1:
                print(f"类别 {current_class} 样本不足，使用均匀权重")
                return np.ones(n_features) / n_features
            # 计算两类均值向量
            mu1 = np.mean(X1, axis=0) # 当前类别均值
            mu0 = np.mean(X0, axis=0) # 其他类别均值
            # 计算类间散度方向
            mean_diff = mu1 - mu0
            # 计算类内散度矩阵 Sw
            if n1 > 1:
                S1 = np.cov(X1.T) * (n1 - 1) # 散度矩阵 = 协方差 × (n-1)
            else:
                S1 = np.zeros((n_features, n_features))
            if n0 > 1:
                S0 = np.cov(X0.T) * (n0 - 1)
            else:
                S0 = np.zeros((n_features, n_features))
            Sw = S1 + S0 # 类内散度矩阵
            # 添加正则化防止矩阵奇异
            epsilon = 1e-6
            Sw_reg = Sw + epsilon * np.eye(n_features)
            # 计算LDA权重向量：w = Sw^{-1} * (mu1 - mu0)
            try:
                # 使用伪逆提高数值稳定性
                Sw_inv = np.linalg.pinv(Sw_reg)
                weights = np.dot(Sw_inv, mean_diff)
            except np.linalg.LinAlgError:
                # 如果伪逆失败，使用简单方法
                print(f"矩阵求逆失败，使用均值差作为权重方向")
                weights = mean_diff
            # 处理可能出现的NaN或Inf值
            weights = np.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
            # 取绝对值并归一化（我们关心特征的重要性大小，不关心方向）
            weights = np.abs(weights)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(n_features) / n_features
            print(f"LDA权重计算完成 - 类别 {current_class}: 权重和={np.sum(weights):.4f}")
            return weights
        except Exception as e:
            print(f"广义瑞利商权重计算错误: {e}")
            n_features = X.shape[1]
            return np.ones(n_features) / n_features
    def _centroid_weights(self, X, y, current_class):
        """
        质心法权重计算：基于每个类别点与整体质心的比例关系
        """
        try:
            # 获取当前类别的样本
            class_mask = (y == current_class)
            X_class = X[class_mask]
            n_samples_class = len(X_class)
            n_features = X.shape[1]
            if n_samples_class == 0:
                print(f"类别 {current_class} 没有样本，使用均匀权重")
                return np.ones(n_features) / n_features
            # 1. 计算整体质心（所有点的均值）
            centroid_all = np.mean(X, axis=0)
            # 避免除零错误，为质心添加小常数
            epsilon = 1e-8
            centroid_all_safe = centroid_all + epsilon
            # 2. 对当前类别的每个点，计算与整体质心的比例
            ratio_vectors = []
            for i in range(n_samples_class):
                x_point = X_class[i]
                # 计算比例：点 / 质心（元素级除法）
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.divide(x_point, centroid_all_safe)
                # 处理除零和无效值
                ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=0.0)
                # 取绝对值，因为我们关心特征的重要性大小
                ratio_abs = np.abs(ratio)
                ratio_vectors.append(ratio_abs)
            # 3. 计算该类所有点比例向量的均值
            if len(ratio_vectors) > 0:
                mean_ratio = np.mean(ratio_vectors, axis=0)
            else:
                mean_ratio = np.ones(n_features)
            # 4. 归一化均值比例向量作为权重
            if np.sum(mean_ratio) > 0:
                weights = mean_ratio / np.sum(mean_ratio)
            else:
                weights = np.ones(n_features) / n_features
            print(f"质心法权重计算完成 - 类别 {current_class}: 样本数={n_samples_class}, 权重和={np.sum(weights):.4f}")
            return weights
        except Exception as e:
            print(f"质心法权重计算错误: {e}")
            n_features = X.shape[1]
            return np.ones(n_features) / n_features
    def _inter_class_difference_weights(self, X, y, current_class):
        try:
            current_class_mask = (y == current_class)
            other_class_mask = (y != current_class)
            X_current = X[current_class_mask]
            X_other = X[other_class_mask]
            if len(X_current) < 2 or len(X_other) < 1:
                n_features = X.shape[1]
                return np.ones(n_features) / n_features
            cov_estimator_current = LedoitWolf().fit(X_current)
            sigma_current = cov_estimator_current.covariance_
            precision_current = np.linalg.pinv(sigma_current)
            mean_current = np.mean(X_current, axis=0)
            mean_other = np.mean(X_other, axis=0)
            mean_diff = mean_current - mean_other
            weights = np.dot(precision_current, mean_diff)
            if np.sum(np.abs(weights)) > 0:
                weights = np.abs(weights) / np.sum(np.abs(weights))
            else:
                n_features = X.shape[1]
                weights = np.ones(n_features) / n_features
            return weights
        except Exception as e:
            print(f"类别间差异权重计算错误: {e}")
            n_features = X.shape[1]
            return np.ones(n_features) / n_features
    def _f_score_weights(self, X, y, current_class):
        try:
            from sklearn.feature_selection import f_classif
            y_binary = (y == current_class).astype(int)
            f_scores, _ = f_classif(X, y_binary)
            f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=1.0, neginf=0.0)
            if np.sum(f_scores) > 0:
                weights = f_scores / np.sum(f_scores)
            else:
                n_features = X.shape[1]
                weights = np.ones(n_features) / n_features
            return weights
        except Exception as e:
            print(f"F-score权重计算错误: {e}")
            n_features = X.shape[1]
            return np.ones(n_features) / n_features
    def _inverse_covariance_weights(self, X, y=None, current_class=None):
        try:
            if current_class is not None and y is not None:
                current_class_mask = (y == current_class)
                X_used = X[current_class_mask]
                if len(X_used) < 2:
                    n_features = X.shape[1]
                    return np.ones(n_features) / n_features
            else:
                X_used = X
            cov_estimator = LedoitWolf().fit(X_used)
            sigma = cov_estimator.covariance_
            precision_matrix = np.linalg.pinv(sigma)
            n_features = X_used.shape[1]
            unit_vector = np.ones(n_features)
            weights = np.dot(precision_matrix, unit_vector)
            if np.sum(np.abs(weights)) > 0:
                weights = np.abs(weights) / np.sum(np.abs(weights))
            else:
                weights = np.ones(n_features) / n_features
            return weights
        except Exception as e:
            print(f"逆协方差权重计算错误: {e}")
            n_features = X.shape[1]
            return np.ones(n_features) / n_features

# ==================== 主函数 ====================
def main():
    """主函数"""
    # 文件路径
    file_path = 'D:/Downloads/archive/winequalityN.csv'
    try:
        # 1. 加载和预处理数据
        print("=== 数据加载和预处理 ===")
        df = pd.read_csv(file_path)
        # 添加行号列（CSV文件中的实际行号，从2开始，因为第一行是标题）
        df['csv_row_number'] = range(2, len(df) + 2)  # 从2开始，因为第一行是标题
        df = df.dropna()
        df['type'] = df['type'].map({'red': 1, 'white': 2, 'Red': 1, 'White': 2}).fillna(0)
        # 分离特征、标签和行号
        # 注意：行号不作为特征使用，只用于追踪
        row_indices = df['csv_row_number'].values
        X = df.drop(['quality', 'csv_row_number'], axis=1).values
        y = df['quality'].values
        # X = df.drop(['species', 'csv_row_number'], axis=1).values
        # y = df['species'].values
        print(f"数据形状: X{X.shape}, y{y.shape}")
        print("类别分布:", Counter(y))
        feature_names = df.drop(['quality', 'csv_row_number'], axis=1).columns.tolist()
        # feature_names = df.drop(['species', 'csv_row_number'], axis=1).columns.tolist()
        print("数据属性:", feature_names)

        # 2. 划分训练测试集
        """划分训练测试集，同时保留行号信息"""
        # 使用相同的随机种子确保划分一致
        X_train, X_test, y_train, y_test, row_indices_train, row_indices_test = train_test_split(
            X, y, row_indices, 
            test_size=0.005, 
            random_state=42, 
            stratify=y
        )
        # 标准化特征
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        minmax_scaler = MinMaxScaler()
        X_train_normalized = minmax_scaler.fit_transform(X_train_scaled)
        X_test_scaled = scaler.fit_transform(X_test)
        X_test_normalized = minmax_scaler.fit_transform(X_test_scaled)
        print(f"数据形状: X{X_train_normalized.shape}, y{y_train.shape}")
        print("类别分布:", Counter(y_train))
        print(f"\n训练集: {X_train.shape}, 测试集: {X_test.shape}")
        # 计算训练集的协方差矩阵
        cov_matrix = np.cov(X_train_normalized.T)  # 转置，因为np.cov期望行是特征，列是样本
        # 计算逆协方差矩阵
        try:
            inv_cov_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # 如果协方差矩阵奇异，使用伪逆
            inv_cov_matrix = np.linalg.pinv(cov_matrix)
        print("逆协方差打印",inv_cov_matrix)
        # 3. 开始测试：选择测试样本并统计正确预测数
        correct_count = 0
        total_count = len(X_test)
        test_result={}
        test_result['details'] = {}
        unfound_count=0
        corrector = BinaryClassWeightCorrector()
        corrector.fit(X_train,y_train)
        local_cov_corrector = ManifoldWeightCorrector()
        local_corrector = BinaryClassWeightCorrector()
        weight_calculator = GlobalWeightMatrix()
        global_weight_calculator = GlobalWeightMatrix()
        for target_idx in range(len(X_test)):
            temp_result= {}
            target_point = X_test[target_idx]
            target_point_normalized = X_test_normalized[target_idx]
            target_class = y_test[target_idx]
            target_point_raw_data = X_train[target_idx]
            target_row = row_indices_test[target_idx]
            # print(f"\n=== 对测试样本进行局部区域搜索 ===")
            # print(f"目标样本真实类别: {target_class}")
            # 4. 使用多策略搜索最佳局部区域
            searcher = MultiStrategyRegionSearcher(inv_cov_matrix=inv_cov_matrix)
            best_region, best_fitness, best_class_dist = searcher.multi_strategy_search(
                X_train_normalized, y_train, target_point_normalized, region_size=15,
            )
            # 5. 分析结果
            # print(f"\n=== 最终搜索结果 ===")
            # print(f"最佳适应度: {best_fitness:.4f}")
            # print(f"局部区域包含样本数: {len(best_region)}")
            print(f"局部区域内类别分布: {dict(best_class_dist)}")
            target_ratio = best_class_dist.get(target_class, 0) / len(best_region)
            # print(f"目标类别在局部区域中的比例: {target_ratio:.4f}")
            # 6. 分析局部区域是否包含目标类别
            if target_class in best_class_dist:
                print("✅ 成功找到包含目标类别的局部区域!")
            else:
                print("❌ 未能找到包含目标类别的局部区域，将使用全局信息")
                unfound_count = unfound_count+1
                best_region, _ = ensure_class_diversity_in_region(X_train_normalized,y_train,target_point_normalized, inv_cov=inv_cov_matrix)
                # total_count = total_count-1
                # 如果未找到目标类别，添加一些目标类别的样本
                # target_class_indices = np.where(y_train == target_class)[0]
                # if len(target_class_indices) > 0:
                #     # 计算目标类别样本与目标点的距离
                #     target_class_points = X_train[target_class_indices]
                #     distances = np.linalg.norm(target_class_points - target_point, axis=1)
                #     # 选择距离最近的目标类别样本
                #     nearest_target_indices = target_class_indices[np.argsort(distances)[:3]] # 添加3个最近的目标类别样本
                #     best_region.extend(nearest_target_indices.tolist())
                #     # 去除重复样本
                #     best_region = list(set(best_region))
                #     # 如果样本数超过，随机删除
                #     if len(best_region) > 15:
                #         best_region = random.sample(best_region, 15)
                #     # 更新类别分布
                #     best_class_dist = Counter(y_train[best_region])
                #     print(f"修正后的类别分布: {dict(best_class_dist)}")
            if best_region is not None:
                print(f"\n" + "="*60)
                print("局部区域搜索完成!")
                print(f"找到了包含 {len(best_region)} 个样本的局部区域")
                print(f"找到了best_region的局部区域 {best_region} ")
                print(f"目标类别 {target_class} 在区域中的比例: {Counter(y_train[best_region]).get(target_class, 0)/len(best_region):.4f}")
                print("="*60)
                
            # 6. 提取局部区域数据
            X_region_row_indices_train = row_indices_train[best_region]
            X_region_raw_data = X_train[best_region]
            X_region = X_train[best_region]
            y_region = y_train[best_region]
            weight_calculator.fit(X_region,y_region)
            global_weight_calculator.fit(X_region)
            X_region = np.array([point - target_point for point in X_region ])
            
            # 7. 分析局部区域内的属性特性(质心法)
            analyzer = LocalRegionAnalyzer()
            # analyzer = CorrectedLocalRegionAnalyzer()
            analysis_results = analyzer.analyze_local_region(X_region, y_region, feature_names)
            # comprehensive_weights = analysis_results['comprehensive_weights']
            # 8. 计算全局信息增益
            global_mi = analyzer.calculate_global_mutual_info(X_region, y_region, feature_names)
            analysis_results['global_mutual_info'] = global_mi
            # 9. 计算综合权重
            comprehensive_weights = analyzer.calculate_comprehensive_weights(
                analysis_results, feature_names
            )
            
            comprehensive_weights = weight_calculator.get_class_weight_dict()
            global_weights = global_weight_calculator.get_class_weight_dict()['global']
            for cls in comprehensive_weights.keys():
                class_weight = comprehensive_weights[cls]
                
                corrected_weight = corrector.correct_weight_vector(class_weight, cls)
                comprehensive_weights[cls] = corrected_weight
            # 显示分析结果的摘要
            # if analysis_results:
            #     print(f"\n=== 分析结果摘要 ===")
            #     for cls in analysis_results['entropy'].keys():
            #         print(f"\n类别 {cls}:")
            #         credibilities = analysis_results['credibility'][cls]
            #         variations = analysis_results['variation'][cls]
            #         mutual_info = analysis_results['mutual_info'][cls]
            #         # 找出可信度最高的属性
            #         max_cred_idx = np.argmax(credibilities)
            #         max_cred_value = credibilities[max_cred_idx]
            #         # 找出信息增量最高的属性
            #         max_mi_idx = np.argmax(mutual_info)
            #         max_mi_value = mutual_info[max_mi_idx]
            #         # 找出离散度最高的属性
            #         max_var_idx = np.argmax(variations)
            #         max_var_value = variations[max_var_idx]
            #         print(f" 属性可信度: '{credibilities}")
            #         print(f" 属性信息增量: '{mutual_info}")
            #         print(f" 属性异变程度: '{variations}")
                    # print(f" 属性融合权重: '{comprehensive_weights[cls]}")
            
            # 10. 使用加权KNN进行分类
            knn_classifier = WeightedKNNClassifier(k=7, inv_cov_matrix= inv_cov_matrix)
            # knn_classifier.fit(X_train, y_train, row_indices_train, feature_names)
            knn_classifier.fit(X_train, y_train, row_indices_train, feature_names)
            # 获取局部区域中包含的类别
            local_region_classes = list(set(y_region))
            # 进行预测
            predicted_class, consistency, knn_results,weighted_votes = knn_classifier.predict_with_distance_weighted_voting(
                target_point, comprehensive_weights, local_region_classes
            )
            # predicted_class, consistency, knn_results = knn_classifier.predict_with_weights(
            #     target_point, global_weights, local_region_classes
            # )
            
            print(f"预测结果 for {target_row}: {'是' if target_class == predicted_class else '否'}")
            # 11. 评估一致性
            true_class_consistency, confidence_scores = knn_classifier.evaluate_consistency(knn_results, target_class)
            if predicted_class==target_class:
                correct_count= correct_count + 1
            # if knn_results:
            #     for cls, result in knn_results.items():
            #         # 输出最终结果
            #         print(f"类别 {cls}: KNN中属于该类别的样本的距离加权投票, 一致性 = {result['consistency']}")
            #     for cls, confidence in confidence_scores.items():
            #         print(f"类别 {cls} 置信度: {confidence:.4f}")
                # print(f"\n=== 最终分类结果 ===")
                # print(f"真实类别: {target_class}")
                # print(f"预测类别: {predicted_class}")
                # print(f"预测是否正确: {'是' if predicted_class == target_class else '否'}")
                # print(f"真实类别在KNN中的一致性: {true_class_consistency:.4f}")
            temp_result['target_point'] = target_point_raw_data
            temp_result['target_row'] = target_row
            temp_result['target_class'] = target_class
            temp_result['region_class_dist'] = best_class_dist
            temp_result['best_region'] = X_region_row_indices_train
            temp_result['X_region_raw_data'] = X_region_raw_data
            temp_result['analysis_results'] = analysis_results
            temp_result['comprehensive_weights'] = comprehensive_weights
            temp_result['knn_results'] = knn_results
            # temp_result['consistency'] = weighted_votes
            temp_result['predicted_class'] = predicted_class
            temp_result['true_class_consistency'] = true_class_consistency
            temp_result['confidence_scores'] = confidence_scores
            temp_result['correctOrNot'] = True if predicted_class == target_class else False
            test_result['details'][target_idx]=temp_result
        test_result['accuracy'] =   correct_count/total_count
        print(f"{total_count}个测试样本测试完成，正确{correct_count}个测试样本，正确率{correct_count/total_count}")
        print(f"仍有{unfound_count}个测试案例没有找到最佳邻域，使用全局信息构建")
        return X_train, y_train,test_result
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None
if __name__ == "__main__":
    X_train, y_train,test_result = main()
    print(f"对测试集的预测情况: ")
    print(f"正确率:{test_result['accuracy']} ")
    print(f"======== 详细测试点数据 ========")
    print(f"运行完毕：58 使用马氏距离")
    # for _, test_details in enumerate(test_result['details'].items()):
    #     test_id=test_details[0]
    #     test_details=test_details[1]
    #     print(f"目标样本id: {test_id}")
    #     print(f"预测是否正确: {'是' if test_details['correctOrNot'] else '否'}")
    #     print(f"目标样本真实类别: {test_details['target_class']}")
    #     print(f"目标样本: {test_details['target_point']}")
    #     print(f"目标样本: {test_details['target_row']}")
    #     print(f"最佳局部: {test_details['best_region']}")
    #     print(f"最佳局部: {test_details['region_class_dist']}")
    #     print(f"最佳局部原始数据: {test_details['X_region_raw_data']}")
    #     print(f"融合权重: {test_details['comprehensive_weights']}")
    #     print(f"真实类别在KNN中的一致性: {test_details['true_class_consistency']}")
    #     print(f"所有类别在KNN结果: {test_details['knn_results']}")
    #     print(f"所有类别在KNN中的一致性: {test_details['consistency']}")
    # print(f"对测试集的预测情况: ")
    # print(f"正确率:{test_result['accuracy']} ")
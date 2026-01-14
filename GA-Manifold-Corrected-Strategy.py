# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 19:20:32 2026

@author: Meimei.Huang
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
from scipy import stats
import random, gc

class SmoothLocalRegionBuilder:
    """
    基于贝叶斯平滑、威尔逊平滑和高斯核的局部区域构建器
    提供平滑的类别代表性评估和距离调整
    """
    def __init__(self, k_region=15, region_size=15, alpha=1.0, beta=1.0, 
                 confidence_level=0.95, sigma=1.0, prior_strength=5):
        self.k_region = k_region # 每个类别的候选集大小
        self.region_size = region_size # 目标局部区域大小
        self.alpha = alpha # 贝叶斯先验参数α
        self.beta = beta # 贝叶斯先验参数β
        self.confidence_level = confidence_level # 威尔逊区间置信水平
        self.sigma = sigma # 高斯核带宽参数
        self.prior_strength = prior_strength # 先验强度
        # 存储中间计算结果
        self.sample_results = {} # 每个样本的详细计算结果
        self.class_tightness = {} # 类别紧度
        self.class_candidates = {} # 类别候选集
    def fit(self, X, y, row_ids, inv_cov_matrix=None):
        self.X = X
        self.y = y
        self.row_ids = row_ids
        self.inv_cov_matrix = inv_cov_matrix
        return self
    def compute_class_tightness(self,class_label, indices):
        """计算类别紧度"""
        class_indices = indices[self.y[indices] == class_label]
        if len(class_indices) == 0:
            return 1.0
        class_samples = self.X[class_indices]
        if len(class_samples) <= 1:
            return 1.0
        centroid = np.mean(class_samples, axis=0)
        avg_distance = np.mean(np.linalg.norm(class_samples - centroid, axis=1))
        tightness = 1.0 / (avg_distance + 1e-8)
        return tightness
    def compute_backward_rank(self, target_point, sample_point, candidate_points):
        """计算向后排名"""
        if len(candidate_points) == 0:
            return 1
        distances_to_sample = [np.linalg.norm(candidate - sample_point) 
                              for candidate in candidate_points]
        target_to_sample_dist = np.linalg.norm(target_point - sample_point)
        all_distances = distances_to_sample + [target_to_sample_dist]
        sorted_indices = np.argsort(all_distances)
        rank = np.where(sorted_indices == len(all_distances) - 1)[0][0] + 1
        return rank
    def bayesian_smoothing(self, k, n, alpha=None, beta=None):
        """
        贝叶斯平滑计算p_bayesian
        k: 成功次数（如排名靠前的计数）
        n: 总试验次数
        """
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if n == 0:
            return 0.5 # 默认先验
        p_bayesian = (k + alpha) / (n + alpha + beta)
        return p_bayesian
    def wilson_score_interval(self, p, n, z_score=None):
        """
        威尔逊得分区间计算p_wilson（置信下限）
        """
        if n == 0:
            return p
        if z_score is None:
            z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        denominator = 1 + z_score**2 / n
        centre = (p + z_score**2 / (2 * n)) / denominator
        width = (z_score * np.sqrt((p * (1 - p)) / n + z_score**2 / (4 * n**2))) / denominator
        p_wilson = centre - width # 使用置信下限作为保守估计
        return max(0, min(1, p_wilson))
    def gaussian_kernel_penalty(self, penalty, sigma=None):
        """高斯核函数计算调整因子"""
        if sigma is None:
            sigma = self.sigma
        adjustment_factor = np.exp(-0.5 * (penalty / sigma) ** 2)
        return adjustment_factor
    def calculate_penalty(self, p_wilson, method='linear'):
        """基于p_wilson计算惩罚项"""
        if method == 'linear':
            penalty = 1.0 - p_wilson # 线性惩罚
        elif method == 'quadratic':
            penalty = (1.0 - p_wilson) ** 2 # 二次惩罚
        elif method == 'sigmoid':
            penalty = 1.0 / (1.0 + np.exp(-10 * (p_wilson - 0.5))) # Sigmoid惩罚
        else:
            penalty = 1.0 - p_wilson
        return penalty
    def compute_representativeness(self, R, r, N, sample_id):
        """
        计算样本的类别代表性（综合贝叶斯平滑、威尔逊平滑和高斯核）
        返回调整因子和所有中间结果
        """
        # 1. 初始代表性评估（基于向后排名）
        initial_representativeness = r # 标准化向后排名作为初始评估
        # 2. 贝叶斯平滑（处理小样本情况）
        k_bayesian = max(1, int(r * N)) # 成功次数：排名靠前的样本
        n_bayesian = N # 总试验次数
        p_bayesian = self.bayesian_smoothing(k_bayesian, n_bayesian)
        # 3. 威尔逊平滑（提供置信区间保守估计）
        p_wilson = self.wilson_score_interval(initial_representativeness, N)
        # 4. 紧度加权调整
        tightness_weight = np.tanh(R - 1) # R>1时正调整，R<1时负调整
        weighted_p_wilson = p_wilson * (1 + 0.3 * tightness_weight)
        weighted_p_wilson = max(0, min(1, weighted_p_wilson))
        # 5. 计算惩罚项
        penalty = self.calculate_penalty(weighted_p_wilson, method='sigmoid')
        # 6. 高斯核调整因子
        adjustment_factor = self.gaussian_kernel_penalty(penalty)
        # 存储所有中间结果
        results = {
            'sample_id': sample_id,
            'R': R,
            'r': r,
            'N': N,
            'initial_representativeness': initial_representativeness,
            'p_bayesian': p_bayesian,
            'p_wilson': p_wilson,
            'weighted_p_wilson': weighted_p_wilson,
            'penalty': penalty,
            'adjustment_factor': adjustment_factor,
            'tightness_weight': tightness_weight
        }
        return adjustment_factor, results
    def build_local_region(self, target_point):
        """
        构建基于平滑代表性评估的局部区域
        """
        print("=== 开始构建平滑局部区域 ===")
        # 重置结果存储
        self.sample_results = {}
        self.class_tightness = {}
        self.class_candidates = {}
        # 获取所有类别
        unique_classes = np.unique(self.y)
        print(f"数据集中包含的类别: {unique_classes}")
        # 步骤1: 为每个类别构建候选集并计算紧度
        candidate_indices = []
        for cls in unique_classes:
            # 获取该类所有样本
            class_indices = np.where(self.y == cls)[0]
            if len(class_indices) == 0:
                continue
            # 计算到目标点的距离
            distances = [np.linalg.norm(self.X[i] - target_point) for i in class_indices]
            # 选择距离最近的k_region个样本
            sorted_indices = np.argsort(distances)[:self.k_region]
            cls_candidates = [class_indices[i] for i in sorted_indices]
            # 计算类别紧度
            tightness = self.compute_class_tightness(cls, class_indices)
            self.class_candidates[cls] = cls_candidates
            self.class_tightness[cls] = tightness
            candidate_indices.extend(cls_candidates)
            print(f"类别 {cls}: 候选集大小 {len(cls_candidates)}, 紧度: {tightness:.4f}")
        # 步骤2: 为每个候选样本计算调整距离
        print("\n=== 计算样本调整距离 ===")
        all_candidates = list(set(candidate_indices))
        adjusted_distances = []
        for candidate_idx in all_candidates:
            cls = self.y[candidate_idx]
            row_id = self.row_ids[candidate_idx]
            if cls not in self.class_candidates:
                continue
            # 获取类别环境
            cls_candidates = self.class_candidates[cls]
            candidate_points = self.X[cls_candidates]
            sample_point = self.X[candidate_idx]
            # 计算原始距离
            original_distance = np.linalg.norm(sample_point - target_point)
            # 计算向后排名
            backward_rank = self.compute_backward_rank(target_point, sample_point, candidate_points)
            # 计算标准化向后排名r
            N = len(cls_candidates)
            if N > 1:
                r = 1 - (backward_rank - 1) / (N - 1)
            else:
                r = 1.0
            # 获取相对紧度R（当前类别紧度/平均紧度）
            avg_tightness = np.mean(list(self.class_tightness.values()))
            R = self.class_tightness[cls] / (avg_tightness + 1e-8)
            # 计算综合代表性调整因子
            adjustment_factor, results = self.compute_representativeness(R, r, N, row_id)
            # 计算调整后距离
            adjusted_distance = original_distance * adjustment_factor
            # 存储完整结果
            results.update({
                'row_id': row_id,
                'class': cls,
                'original_distance': original_distance,
                'adjusted_distance': adjusted_distance,
                'backward_rank': backward_rank,
                'R': R,
                'r': r
            })
            self.sample_results[row_id] = results
            adjusted_distances.append((candidate_idx, adjusted_distance, cls, row_id))
            # 打印前几个样本的详细结果
            if len(self.sample_results) <= 5:
                print(f"样本 {row_id} (类 {cls}): "
                      f"原始距离={original_distance:.4f}, 调整后={adjusted_distance:.4f}, "
                      f"调整因子={adjustment_factor:.4f}")
        # 步骤3: 按类别比例选择样本构建局部区域
        print("\n=== 按类别比例构建局部区域 ===")
        local_region = []
        class_distribution = Counter(self.y)
        total_samples = sum(class_distribution.values())
        # 计算每个类别的目标样本数
        class_targets = {}
        for cls, count in class_distribution.items():
            target_count = max(1, int(self.region_size * count / total_samples))
            class_targets[cls] = min(target_count, len(self.class_candidates.get(cls, [])))
        print(f"类别目标分布: {class_targets}")
        # 按类别选择调整距离最小的样本
        for cls, target_count in class_targets.items():
            if cls not in self.class_candidates or target_count == 0:
                continue
            # 获取该类别候选样本
            cls_candidates = [idx for idx, _, cls_val, row_id in adjusted_distances if cls_val == cls]
            if not cls_candidates:
                continue
            # 按调整距离排序
            cls_candidates_sorted = sorted(
                cls_candidates,
                key=lambda idx: next(adj_dist for i, adj_dist, c, rid in adjusted_distances 
                                   if i == idx and c == cls)
            )[:target_count]
            local_region.extend(cls_candidates_sorted)
            print(f"类别 {cls}: 选择 {len(cls_candidates_sorted)} 个样本")
        # 步骤4: 如果区域大小不足，补充调整距离最小的样本
        if len(local_region) < self.region_size:
            remaining = self.region_size - len(local_region)
            used_set = set(local_region)
            available_candidates = [idx for idx, _, _, _ in adjusted_distances 
                                  if idx not in used_set]
            if available_candidates:
                available_sorted = sorted(
                    available_candidates,
                    key=lambda idx: next(adj_dist for i, adj_dist, c, rid in adjusted_distances 
                                       if i == idx)
                )[:remaining]
                local_region.extend(available_sorted)
                print(f"补充 {len(available_sorted)} 个样本")
        # 最终调整大小
        local_region = local_region[:self.region_size]
        # 步骤5: 输出调试信息
        self._print_debug_info(local_region)
        return local_region
    def _print_debug_info(self, local_region):
        """打印详细的调试信息"""
        print("\n" + "="*60)
        print("调试信息 - 样本代表性分析")
        print("="*60)
        # 获取局部区域中样本的row_id
        local_region_row_ids = [self.row_ids[idx] for idx in local_region]
        # 按p_bayesian排序
        print("\n1. 按p_bayesian排序 (贝叶斯平滑代表性):")
        sorted_bayesian = sorted(self.sample_results.items(), 
                               key=lambda x: x[1]['p_bayesian'], reverse=True)
        for i, (row_id, results) in enumerate(sorted_bayesian[:5]):
            if row_id in local_region_row_ids:
                marker = "✓"
            else:
                marker = " "
            print(f"{marker} 排名{i+1}: 样本{row_id} - p_bayesian={results['p_bayesian']:.4f}, "
                  f"类{results['class']}")
        # 按p_wilson排序
        print("\n2. 按p_wilson排序 (威尔逊平滑代表性):")
        sorted_wilson = sorted(self.sample_results.items(), 
                             key=lambda x: x[1]['p_wilson'], reverse=True)
        for i, (row_id, results) in enumerate(sorted_wilson[:5]):
            if row_id in local_region_row_ids:
                marker = "✓"
            else:
                marker = " "
            print(f"{marker} 排名{i+1}: 样本{row_id} - p_wilson={results['p_wilson']:.4f}, "
                  f"类{results['class']}")
        # 按penalty排序
        print("\n3. 按penalty排序 (惩罚值，越小越好):")
        sorted_penalty = sorted(self.sample_results.items(), 
                              key=lambda x: x[1]['penalty'])
        for i, (row_id, results) in enumerate(sorted_penalty[:5]):
            if row_id in local_region_row_ids:
                marker = "✓"
            else:
                marker = " "
            print(f"{marker} 排名{i+1}: 样本{row_id} - penalty={results['penalty']:.4f}, "
                  f"类{results['class']}")
        # 按调整因子排序
        print("\n4. 按调整因子排序 (距离缩放因子，越小越好):")
        sorted_factor = sorted(self.sample_results.items(), 
                             key=lambda x: x[1]['adjustment_factor'])
        for i, (row_id, results) in enumerate(sorted_factor[:5]):
            if row_id in local_region_row_ids:
                marker = "✓"
            else:
                marker = " "
            print(f"{marker} 排名{i+1}: 样本{row_id} - 调整因子={results['adjustment_factor']:.4f}, "
                  f"类{results['class']}")
        # 局部区域统计
        local_classes = [self.y[idx] for idx in local_region]
        class_dist = Counter(local_classes)
        print(f"\n5. 局部区域最终统计:")
        print(f"区域大小: {len(local_region)}")
        print(f"类别分布: {dict(class_dist)}")
        # 计算平均调整因子
        avg_adjustment = np.mean([self.sample_results[self.row_ids[idx]]['adjustment_factor'] 
                                for idx in local_region])
        print(f"平均调整因子: {avg_adjustment:.4f}")
    def get_detailed_results(self, sort_by='p_bayesian'):
        """获取详细结果并排序"""
        if not self.sample_results:
            return []
        if sort_by == 'p_bayesian':
            sorted_results = sorted(self.sample_results.items(), 
                                  key=lambda x: x[1]['p_bayesian'], reverse=True)
        elif sort_by == 'p_wilson':
            sorted_results = sorted(self.sample_results.items(), 
                                  key=lambda x: x[1]['p_wilson'], reverse=True)
        elif sort_by == 'penalty':
            sorted_results = sorted(self.sample_results.items(), 
                                  key=lambda x: x[1]['penalty'])
        elif sort_by == 'adjustment_factor':
            sorted_results = sorted(self.sample_results.items(), 
                                  key=lambda x: x[1]['adjustment_factor'])
        elif sort_by == 'adjusted_distance':
            sorted_results = sorted(self.sample_results.items(), 
                                  key=lambda x: x[1]['adjusted_distance'])
        else:
            sorted_results = list(self.sample_results.items())
        return sorted_results
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

# ==================== 加权KNN分类器 ====================
class WeightedKNNClassifier:
    """基于局部区域权重的加权KNN分类器"""
    def __init__(self, k=7, distance_weight_method='exponential', inv_cov_matrix=None):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.X_row_number = None
        self.feature_names = None
        self.distance_weight_method = distance_weight_method
        self.inv_cov_matrix = inv_cov_matrix
    def fit(self, X_train, y_train, X_row_number, feature_names=None):
        """训练模型"""
        self.X_train = X_train
        self.y_train = y_train
        self.X_row_number = X_row_number
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X_train.shape[1])]
        print(f"训练集大小: {X_train.shape}")
    def calculate_weighted_distance(self, target_point, weights):
        """计算加权欧式/马氏距离"""
        # 确保权重是归一化的
        weights = weights / np.sum(weights)
        # 计算加权距离
        weighted_distances = []
        for i in range(len(self.X_train)):
            # 特征加权欧氏距离
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
    def predict_with_weights(self, target_point, comprehensive_weights, local_region_classes):
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
        for cls, weights in comprehensive_weights.items():
            if cls not in local_region_classes:
                print(f"跳过类别 {cls}，因为不在局部区域中")
                continue
            # print(f"\n使用类别 {cls} 的权重计算距离...")
            # print(f"权重向量: {[f'{w:.4f}' for w in weights]}")
            # 计算加权距离
            weighted_distances = self.calculate_weighted_distance(target_point, weights)
            # 按距离排序，选择前k个邻居
            weighted_distances.sort(key=lambda x: x[1])
            k_neighbors = weighted_distances[:self.k]
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
    def predict_with_voting(self, target_point, comprehensive_weights, local_region_classes):
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
            class_count = sum(1 for _, _, neighbor_class in k_neighbors if neighbor_class == cls)
            consistency = class_count / self.k
            
            # 计算加权投票（考虑距离权重）
            weighted_vote = 0.0
            for rank, neighbor_class in enumerate(neighbor_classes):
                if neighbor_class == cls:
                    # weighted_vote = 1/(2**rank)
                    # weighted_vote += 1/(distances[rank]+1e-8)
                    weighted_vote =consistency * 1/(distances[rank]+1e-8)
                    # weighted_vote = 1/(distances[rank]+1e-8)
                    # 对于wine数据集不要加 consistency
                    break
            weighted_consistency = weighted_vote
            weighted_votes[cls] = weighted_consistency
            
            print(f"\n类别 {cls} 加权投票权重是：",weighted_consistency)
            # 存储详细结果
            knn_results[cls] = {
                'neighbors': k_neighbors,
                'distances': distances,
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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
class MemoryEfficientKNNVisualizer:
    """内存高效的KNN分类器可视化工具"""
    
    def __init__(self, classifier, feature_names=None):
        self.classifier = classifier
        self.feature_names = feature_names
    def fit(self, X, y):
        self.X = X
        self.y = y
        return self
    def plot_decision_boundary_2d_optimized(self,target_point, comprehensive_weights, 
                                           local_region_classes, method='pca', figsize=(12, 8), 
                                           resolution_ratio=0.1):
        """
        内存优化的二维决策边界绘制
        resolution_ratio: 分辨率比例因子，降低网格密度
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 降维处理
        if method == 'pca':
            reducer = PCA(n_components=2)
            reducer_name = "PCA"
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.X)-1))
            reducer_name = "t-SNE"
        
        X_reduced = reducer.fit_transform(self.X)
        target_reduced = reducer.transform(target_point.reshape(1, -1))
        
        # 动态调整网格分辨率
        x_range = X_reduced[:, 0].max() - X_reduced[:, 0].min()
        y_range = X_reduced[:, 1].max() - X_reduced[:, 1].min()
        
        # 根据数据范围自适应调整步长
        adaptive_step_x = max(0.02, x_range * resolution_ratio)
        adaptive_step_y = max(0.02, y_range * resolution_ratio)
        
        x_min, x_max = X_reduced[:, 0].min() - 0.1, X_reduced[:, 0].max() + 0.1
        y_min, y_max = X_reduced[:, 1].min() - 0.1, X_reduced[:, 1].max() + 0.1
        
        # 使用稀疏网格点
        xx, yy = np.meshgrid(np.arange(x_min, x_max, adaptive_step_x),
                           np.arange(y_min, y_max, adaptive_step_y))
        
        print(f"网格大小: {xx.shape}, 预计内存占用: {xx.nbytes / 1024 / 1024:.2f} MB")
        
        # 分批处理网格预测，避免内存峰值
        Z = self._batch_predict_mesh(reducer, xx, yy, comprehensive_weights, 
                                   local_region_classes, batch_size=1000)
        
        # 绘制决策边界
        contour = axes[0].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu', levels=20)
        scatter = axes[0].scatter(X_reduced[:, 0], X_reduced[:, 1], c=self.y, cmap='viridis', 
                                alpha=0.6, s=30)
        axes[0].scatter(target_reduced[0, 0], target_reduced[0, 1], 
                       c='red', marker='*', s=200, label='Test Point')
        axes[0].set_title(f'optimisied decision boundary \n({reducer_name} projection)')
        axes[0].set_xlabel(f'{reducer_name} Component 1')
        axes[0].set_ylabel(f'{reducer_name} Component 2')
        
        # 添加颜色条
        plt.colorbar(contour, ax=axes[0])
        
        # 第二子图：简化版本，只显示样本分布
        self._plot_simple_distribution(X_reduced, self.y, target_reduced, axes[1], reducer_name)
        
        plt.tight_layout()
        return fig
    
    def _batch_predict_mesh(self, reducer, xx, yy, comprehensive_weights, 
                          local_region_classes, batch_size=1000):
        """分批处理网格预测，减少内存峰值使用"""
        Z = np.zeros(xx.ravel().shape)
        total_points = len(xx.ravel())
        
        for i in range(0, total_points, batch_size):
            end_idx = min(i + batch_size, total_points)
            batch_points = np.c_[xx.ravel()[i:end_idx], yy.ravel()[i:end_idx]]
            
            # 逆变换回原始空间（近似）
            try:
                # 尝试使用逆变换，如果不可用则使用最近邻近似
                batch_original = reducer.inverse_transform(batch_points)
            except AttributeError:
                # t-SNE等没有逆变换，使用最近邻近似
                batch_original = self._approximate_inverse_transform(reducer, batch_points)
            
            # 对每个点进行预测
            for j, point in enumerate(batch_original):
                Z[i + j] = self._predict_single_point(point, comprehensive_weights, 
                                                     local_region_classes)
            
            # 手动触发垃圾回收[1]
            if i % (batch_size * 10) == 0:
                gc.collect()
                
            print(f"处理进度: {end_idx}/{total_points} ({end_idx/total_points*100:.1f}%)")
        
        return Z.reshape(xx.shape)
    
    def _approximate_inverse_transform(self, reducer, points):
        """近似逆变换（用于t-SNE等没有逆变换的降维方法）"""
        # 使用最近邻方法近似逆变换
        from sklearn.neighbors import NearestNeighbors
        
        # 获取降维前的训练数据
        if hasattr(reducer, 'embedding_'):
            X_original = reducer._X
            X_reduced = reducer.embedding_
        else:
            # 如果没有嵌入数据，使用拟合数据
            X_original = reducer._X
            X_reduced = reducer.transform(X_original)
        
        # 找到最近邻
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_reduced)
        distances, indices = nbrs.kneighbors(points)
        
        return X_original[indices.flatten()]
    
    def _predict_single_point(self, point, comprehensive_weights, local_region_classes):
        """对单个点进行预测（简化版，避免完整KNN计算）"""
        try:
            # 使用加权距离计算预测得分
            scores = {}
            for cls, weights in comprehensive_weights.items():
                if cls not in local_region_classes:
                    continue
                
                # 简化距离计算
                weighted_dist = np.linalg.norm(weights * (self.classifier.X_train - point), axis=1)
                min_dist = np.min(weighted_dist)
                scores[cls] = 1.0 / (min_dist + 1e-8)
            
            if scores:
                return max(scores.items(), key=lambda x: x[1])[0]
            else:
                return 0
        except:
            return 0
    
    def _plot_simple_distribution(self, X_reduced, y, target_reduced, ax, reducer_name):
        """绘制简化的样本分布图"""
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', 
                           alpha=0.6, s=20)
        ax.scatter(target_reduced[0, 0], target_reduced[0, 1], 
                  c='red', marker='*', s=150, label='Test Point')
        ax.set_title(f'sample distribution ({reducer_name})')
        ax.set_xlabel(f'{reducer_name} component 1')
        ax.set_ylabel(f'{reducer_name} component 2')
        ax.legend()
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax)
    
    def plot_alternative_visualizations(self, X, y, comprehensive_weights, 
                                      knn_results, weighted_votes, true_class=None):
        """
        提供替代的可视化方案，避免高内存使用
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 特征重要性热力图
        self._plot_feature_heatmap(comprehensive_weights, axes[0, 0])
        
        # 2. 投票结果条形图
        self._plot_voting_bars(weighted_votes, true_class, axes[0, 1])
        
        # 3. 距离分布直方图
        self._plot_distance_histogram(knn_results, axes[1, 0])
        
        # 4. 类别分布饼图
        self._plot_class_distribution(y, axes[1, 1])
        
        plt.tight_layout()
        return fig
    
    def _plot_feature_heatmap(self, comprehensive_weights, ax):
        """绘制特征重要性热力图"""
        weights_matrix = np.array([comprehensive_weights[cls] 
                                 for cls in sorted(comprehensive_weights.keys())])
        
        im = ax.imshow(weights_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_title("特征权重热力图")
        ax.set_xticks(range(len(self.feature_names)))
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax.set_yticks(range(len(comprehensive_weights)))
        ax.set_yticklabels([f'Class {cls}' for cls in sorted(comprehensive_weights.keys())])
        
        plt.colorbar(im, ax=ax)
    
    def _plot_voting_bars(self, weighted_votes, true_class, ax):
        """绘制投票结果条形图"""
        classes = list(weighted_votes.keys())
        votes = list(weighted_votes.values())
        
        colors = ['lightgreen' if cls == true_class else 'lightblue' for cls in classes]
        bars = ax.bar(classes, votes, color=colors, alpha=0.7)
        
        ax.set_title("加权投票结果")
        ax.set_ylabel("投票得分")
        ax.set_xlabel("类别")
        
        # 添加数值标签
        for bar, vote in zip(bars, votes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{vote:.4f}', ha='center', va='bottom')
    
    def _plot_distance_histogram(self, knn_results, ax):
        """绘制距离分布直方图"""
        all_distances = []
        for cls, result in knn_results.items():
            distances = result['distances']
            all_distances.extend(distances)
            ax.hist(distances, alpha=0.6, label=f'Class {cls}', bins=15)
        
        ax.set_title("各类别距离分布")
        ax.set_xlabel("距离")
        ax.set_ylabel("频次")
        ax.legend()
    
    def _plot_class_distribution(self, y, ax):
        """绘制类别分布饼图"""
        class_counts = Counter(y)
        ax.pie(class_counts.values(), labels=[f'Class {cls}' for cls in class_counts.keys()],
               autopct='%1.1f%%', startangle=90)
        ax.set_title("类别分布")
class LightweightKNNVisualizer:
    """修正的轻量级KNN可视化工具，修复了索引错误"""
    def __init__(self, classifier, feature_names=None):
        self.classifier = classifier
        self.feature_names = feature_names
    def fit(self, X, y):
        """存储训练数据"""
        self.X = X
        self.y = y
        return self
    def _safe_pca_transform(self, data, n_components=2):
        """安全执行PCA转换，处理内存问题"""
        try:
            pca = PCA(n_components=min(n_components, data.shape[0], data.shape[1]))
            return pca.fit_transform(data)
        except Exception as e:
            print(f"PCA转换失败: {e}, 使用随机投影")
            # 如果PCA失败，使用简单的随机投影
            np.random.seed(42)
            proj = np.random.randn(data.shape[1], n_components)
            proj /= np.linalg.norm(proj, axis=0)
            return data @ proj
    def _get_local_points(self, target_point, k_per_class=20, max_total_points=1000):
        """获取局部点，但限制总点数"""
        local_points = []
        local_labels = []
        # 计算所有点到目标点的距离
        distances = np.linalg.norm(self.X - target_point, axis=1)
        # 获取所有类别
        unique_classes = np.unique(self.y)
        for cls in unique_classes:
            # 获取该类别的索引
            class_indices = np.where(self.y == cls)[0]
            if len(class_indices) == 0:
                continue
            # 计算该类中每个样本到测试点的距离
            class_distances = distances[class_indices]
            # 找出最近的k个样本
            k = min(k_per_class, len(class_indices))
            nearest_indices = np.argsort(class_distances)[:k]
            # 收集局部点
            local_points.append(self.X[class_indices[nearest_indices]])
            local_labels.append(self.y[class_indices[nearest_indices]])
        # 合并所有点
        if local_points:
            all_points = np.vstack(local_points)
            all_labels = np.hstack(local_labels)
            # 如果总点数太多，随机采样
            if len(all_points) > max_total_points:
                indices = np.random.choice(len(all_points), max_total_points, replace=False)
                all_points = all_points[indices]
                all_labels = all_labels[indices]
        else:
            all_points = np.array([])
            all_labels = np.array([])
        return all_points, all_labels
    def plot_focused_local_boundary(self, target_point, comprehensive_weights, 
                                   k_per_class=20, method='pca', figsize=(12, 8)):
        """
        聚焦的局部决策边界可视化
        只显示测试点周围的k个最近邻，不绘制整个网格
        """
        # 获取局部点
        local_points, local_labels = self._get_local_points(
            target_point, k_per_class, max_total_points=200
        )
        if len(local_points) == 0:
            print("警告: 没有找到局部样本")
            return None
        # 将目标点添加到局部点中
        all_points = np.vstack([local_points, target_point.reshape(1, -1)])
        all_labels = np.append(local_labels, -1) # 用-1表示目标点
        # 降维
        if method == 'pca':
            reduced_points = self._safe_pca_transform(all_points)
        else:
            # 使用t-SNE，但对于大量数据可能需要调整
            if len(all_points) > 1000:
                # 使用子采样
                indices = np.random.choice(len(all_points), 1000, replace=False)
                tsne_points = all_points[indices]
                tsne_labels = all_labels[indices]
                contains_target = -1 in tsne_labels
                if not contains_target:
                    # 确保目标点包含在内
                    tsne_points = np.vstack([tsne_points, target_point.reshape(1, -1)])
                    tsne_labels = np.append(tsne_labels, -1)
                reducer = TSNE(n_components=2, random_state=42, 
                             perplexity=min(30, len(tsne_points)-1))
                reduced_subset = reducer.fit_transform(tsne_points)
                # 重建完整的降维点
                reduced_points = np.zeros((len(all_points), 2))
                reduced_points[indices] = reduced_subset[:-1] if contains_target else reduced_subset
                if not contains_target:
                    reduced_points[-1] = reduced_subset[-1]
            else:
                reducer = TSNE(n_components=2, random_state=42, 
                             perplexity=min(30, len(all_points)-1))
                reduced_points = reducer.fit_transform(all_points)
        # 分离目标点和其他点
        target_idx = -1
        test_point_reduced = reduced_points[target_idx]
        other_points_reduced = reduced_points[:-1]
        other_labels = all_labels[:-1]
        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # 1. 局部点散点图
        self._plot_local_scatter(axes[0], other_points_reduced, other_labels, 
                               test_point_reduced, method)
        # 2. 修正的距离热力图
        self._plot_distance_heatmap_fixed(axes[1], target_point, local_points, 
                                         local_labels, method)
        # 3. Voronoi图（近似决策边界）
        self._plot_voronoi_approximation(axes[2], other_points_reduced, other_labels, 
                                       test_point_reduced, method)
        plt.suptitle(f'Local Decision Region Analysis (k={k_per_class} per class)', fontsize=16)
        plt.tight_layout()
        return fig
    def _plot_local_scatter(self, ax, points, labels, test_point, method):
        """绘制局部点的散点图"""
        # 获取唯一标签
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        # 为每个类别绘制点
        for i, label in enumerate(unique_labels):
            if label == -1: # 跳过目标点
                continue
            mask = (labels == label)
            ax.scatter(points[mask, 0], points[mask, 1], 
                      c=[colors[i]], label=f'Class {label}',
                      alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        # 绘制测试点
        ax.scatter(test_point[0], test_point[1], c='red', marker='*', 
                  s=300, label='Test Point', edgecolors='black', linewidth=1.5, zorder=10)
        ax.set_xlabel(f'{method} Component 1')
        ax.set_ylabel(f'{method} Component 2')
        ax.set_title('Local Neighborhood Points')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    def _plot_distance_heatmap_fixed(self, ax, target_point, local_points, 
                                    local_labels, method):
        """修正的距离热力图绘制，避免索引错误"""
        from scipy.spatial.distance import pdist, squareform
        if len(local_points) < 2:
            ax.text(0.5, 0.5, 'Not enough points\nfor heatmap', 
                   ha='center', va='center')
            ax.set_title('Distance Heatmap (Insufficient Data)')
            return
        try:
            # 创建包含所有点的数组（局部点 + 目标点）
            all_points = np.vstack([local_points, target_point.reshape(1, -1)])
            # 如果点太多，采样一部分
            max_points_for_heatmap = 50 # 限制热力图的点数
            if len(all_points) > max_points_for_heatmap:
                # 确保采样时包括目标点
                n_points = min(max_points_for_heatmap, len(all_points))
                # 确保至少有一个目标点
                n_local_points = n_points - 1
                # 从局部点中随机采样
                if n_local_points > 0 and len(local_points) > 0:
                    local_indices = np.random.choice(len(local_points), 
                                                    min(n_local_points, len(local_points)), 
                                                    replace=False)
                    sampled_local_points = local_points[local_indices]
                    sampled_local_labels = local_labels[local_indices]
                else:
                    sampled_local_points = np.array([])
                    sampled_local_labels = np.array([])
                # 添加目标点
                sampled_points = np.vstack([sampled_local_points, target_point.reshape(1, -1)])
                # 创建标签数组
                if len(sampled_local_labels) > 0:
                    # 将数值标签转换为字符串
                    label_strs = [f'C{int(l)}' for l in sampled_local_labels]
                    sampled_labels = np.append(label_strs, ['Target'])
                else:
                    sampled_labels = ['Target']
            else:
                # 使用所有点
                sampled_points = all_points
                if len(local_labels) > 0:
                    label_strs = [f'C{int(l)}' for l in local_labels]
                    sampled_labels = np.append(label_strs, ['Target'])
                else:
                    sampled_labels = ['Target']
            # 计算距离矩阵
            if len(sampled_points) > 1:
                distances = pdist(sampled_points, metric='euclidean')
                distance_matrix = squareform(distances)
                # 绘制热力图
                im = ax.imshow(distance_matrix, cmap='YlOrRd', aspect='auto')
                # 设置刻度
                n_ticks = len(sampled_labels)
                ax.set_xticks(range(n_ticks))
                ax.set_yticks(range(n_ticks))
                # 设置刻度标签
                if n_ticks <= 20: # 如果点数不多，显示所有标签
                    ax.set_xticklabels(sampled_labels, rotation=45, ha='right', fontsize=8)
                    ax.set_yticklabels(sampled_labels, fontsize=8)
                else: # 点数太多，只显示部分标签
                    tick_indices = np.linspace(0, n_ticks-1, 10, dtype=int)
                    ax.set_xticks(tick_indices)
                    ax.set_yticks(tick_indices)
                    ax.set_xticklabels([sampled_labels[i] for i in tick_indices], 
                                      rotation=45, ha='right', fontsize=8)
                    ax.set_yticklabels([sampled_labels[i] for i in tick_indices], 
                                      fontsize=8)
                ax.set_xlabel('Points')
                ax.set_ylabel('Points')
                ax.set_title(f'Distance Heatmap ({len(sampled_points)} points)')
                # 添加颜色条
                plt.colorbar(im, ax=ax, label='Distance')
            else:
                ax.text(0.5, 0.5, 'Not enough points\nfor heatmap', 
                       ha='center', va='center')
                ax.set_title('Distance Heatmap (Insufficient Data)')
        except Exception as e:
            print(f"绘制热力图时出错: {e}")
            ax.text(0.5, 0.5, f'Error creating heatmap\n{str(e)[:50]}...', 
                   ha='center', va='center')
            ax.set_title('Distance Heatmap (Error)')
    def _plot_voronoi_approximation(self, ax, points, labels, test_point, method):
        """绘制Voronoi图近似决策边界"""
        from scipy.spatial import Voronoi, voronoi_plot_2d
        if len(points) < 3:
            ax.text(0.5, 0.5, 'Need at least 3 points\nfor Voronoi diagram', 
                   ha='center', va='center')
            ax.set_title('Voronoi Regions (Insufficient Data)')
            return
        try:
            # 计算Voronoi图
            vor = Voronoi(points)
            # 绘制Voronoi图
            voronoi_plot_2d(vor, ax=ax, show_points=True, show_vertices=True, line_colors='blue',
                          line_width=1, line_alpha=0.6, point_size=20)
            # 用不同颜色标记不同类别的点
            unique_labels = np.unique(labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            for i, label in enumerate(unique_labels):
                if label == -1: # 跳过目标点
                    continue
                mask = (labels == label)
                ax.scatter(points[mask, 0], points[mask, 1], 
                          c=[colors[i]], label=f'Class {label}',
                          alpha=0.8, s=50, edgecolors='black', linewidth=1, zorder=5)
            # 标记测试点
            ax.scatter(test_point[0], test_point[1], c='red', marker='*', 
                      s=200, label='Test Point', edgecolors='black', linewidth=2, zorder=10)
            ax.set_xlabel(f'{method} Component 1')
            ax.set_ylabel(f'{method} Component 2')
            ax.set_title('Voronoi Regions (Decision Boundary Approx.)')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'Voronoi error: {str(e)[:50]}...', 
                   ha='center', va='center')
            ax.set_title('Voronoi Regions (Error)')
    def plot_simplified_local_view(self, target_point, k_neighbors=50, method='pca', figsize=(12, 8)):
        """
        简化的局部视图：只显示最近邻和距离
        完全避免复杂计算
        """
        # 找到k个最近邻
        distances = np.linalg.norm(self.X - target_point, axis=1)
        nearest_indices = np.argsort(distances)[:k_neighbors]
        # 获取最近邻点
        nearest_points = self.X[nearest_indices]
        nearest_labels = self.y[nearest_indices]
        nearest_distances = distances[nearest_indices]
        # 使用PCA降维
        all_points = np.vstack([nearest_points, target_point.reshape(1, -1)])
        if method == 'pca':
            reducer = PCA(n_components=2)
            reduced_points = reducer.fit_transform(all_points)
        else:
            reducer = TSNE(n_components=2, random_state=42, 
                          perplexity=min(30, len(all_points)-1))
            reduced_points = reducer.fit_transform(all_points)
        # 分离目标点
        test_point_reduced = reduced_points[-1]
        neighbor_points_reduced = reduced_points[:-1]
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        # 左图：最近邻散点图
        unique_labels = np.unique(nearest_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = (nearest_labels == label)
            if np.sum(mask) > 0:
                axes[0].scatter(neighbor_points_reduced[mask, 0], neighbor_points_reduced[mask, 1],
                              c=[colors[i]], label=f'Class {label}', alpha=0.7, s=50)
        # 绘制测试点
        axes[0].scatter(test_point_reduced[0], test_point_reduced[1], c='red', marker='*',
                       s=300, label='Test Point', edgecolors='black', linewidth=2, zorder=10)
        # 连接测试点到最近邻
        for i, (x, y) in enumerate(neighbor_points_reduced):
            axes[0].plot([test_point_reduced[0], x], [test_point_reduced[1], y], 
                        'gray', alpha=0.3, linewidth=0.5)
        axes[0].set_xlabel(f'{method} Component 1')
        axes[0].set_ylabel(f'{method} Component 2')
        axes[0].set_title(f'Top {k_neighbors} Nearest Neighbors')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        # 右图：距离条形图
        sorted_indices = np.argsort(nearest_distances)
        sorted_distances = nearest_distances[sorted_indices]
        sorted_labels = nearest_labels[sorted_indices]
        # 为每个类别使用不同颜色
        for i, label in enumerate(unique_labels):
            mask = (sorted_labels == label)
            axes[1].bar(np.where(mask)[0], sorted_distances[mask], 
                       color=colors[i], alpha=0.7, label=f'Class {label}')
        axes[1].set_xlabel('Neighbor Index (sorted by distance)')
        axes[1].set_ylabel('Distance to Test Point')
        axes[1].set_title('Distance to Nearest Neighbors')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig
class WeightedKNNVisualizer:
    """基于属性权重的KNN可视化工具，展示加权距离下的分类边界"""
    def __init__(self, classifier, feature_names=None):
        self.classifier = classifier
        self.feature_names = feature_names
    def fit(self, X, y):
        """存储训练数据"""
        self.X = X
        self.y = y
        return self
    def _calculate_weighted_distance(self, point1, point2, weights):
        """计算加权欧氏距离"""
        # 确保权重是归一化的
        if np.sum(weights) != 1.0:
            weights = weights / np.sum(weights)
        # 计算加权距离
        weighted_diff = weights * (point1 - point2)
        distance = np.sqrt(np.sum(weighted_diff ** 2))
        return distance
    def _get_weighted_local_points(self, target_point, weight_vector, k_per_class=20, max_total_points=1000):
        """获取基于加权距离的局部点"""
        local_points = []
        local_labels = []
        weighted_distances = []
        # 计算所有点到目标点的加权距离
        distances = []
        for i in range(len(self.X)):
            dist = self._calculate_weighted_distance(self.X[i], target_point, weight_vector)
            distances.append((i, dist))
        # 按加权距离排序
        distances.sort(key=lambda x: x[1])
        # 获取所有类别
        unique_classes = np.unique(self.y)
        for cls in unique_classes:
            # 获取该类别的所有样本
            class_indices = np.where(self.y == cls)[0]
            if len(class_indices) == 0:
                continue
            # 找出该类中距离最近的k个样本
            class_distances = [(idx, dist) for idx, dist in distances if idx in class_indices]
            k = min(k_per_class, len(class_distances))
            nearest_class_indices = [idx for idx, dist in class_distances[:k]]
            # 收集局部点
            local_points.append(self.X[nearest_class_indices])
            local_labels.append(self.y[nearest_class_indices])
            # 存储加权距离
            for idx in nearest_class_indices:
                dist = next(dist for i, dist in distances if i == idx)
                weighted_distances.append(dist)
        # 合并所有点
        if local_points:
            all_points = np.vstack(local_points)
            all_labels = np.hstack(local_labels)
            # 如果总点数太多，随机采样
            if len(all_points) > max_total_points:
                indices = np.random.choice(len(all_points), max_total_points, replace=False)
                all_points = all_points[indices]
                all_labels = all_labels[indices]
                weighted_distances = [weighted_distances[i] for i in indices]
        else:
            all_points = np.array([])
            all_labels = np.array([])
            weighted_distances = []
        return all_points, all_labels, weighted_distances
    def plot_weighted_decision_boundary(self, target_point, weight_vector, 
                                      k_per_class=20, method='pca', figsize=(15, 10)):
        """
        绘制基于属性权重的决策边界
        weight_vector: 测试点真实类别的属性权重向量
        """
        # 获取基于加权距离的局部点
        local_points, local_labels, weighted_distances = self._get_weighted_local_points(
            target_point, weight_vector, k_per_class, max_total_points=200
        )
        if len(local_points) == 0:
            print("警告: 没有找到局部样本")
            return None
        # 将目标点添加到局部点中
        all_points = np.vstack([local_points, target_point.reshape(1, -1)])
        all_labels = np.append(local_labels, -1) # 用-1表示目标点
        # 降维
        if method == 'pca':
            reducer = PCA(n_components=2)
            reduced_points = reducer.fit_transform(all_points)
            reducer_name = "PCA"
        else:
            reducer = TSNE(n_components=2, random_state=42, 
                          perplexity=min(30, len(all_points)-1))
            reduced_points = reducer.fit_transform(all_points)
            reducer_name = "t-SNE"
        # 分离目标点和其他点
        test_point_reduced = reduced_points[-1]
        neighbor_points_reduced = reduced_points[:-1]
        neighbor_labels = all_labels[:-1]
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        # 1. 加权距离下的局部点散点图
        self._plot_weighted_scatter(axes[0, 0], neighbor_points_reduced, neighbor_labels, 
                                  test_point_reduced, weighted_distances, reducer_name)
        # 2. 基于加权距离的决策边界
        self._plot_weighted_decision_surface(axes[0, 1], local_points, local_labels, 
                                           target_point, weight_vector, reducer, reducer_name)
        # 3. 属性权重可视化
        self._plot_feature_weights(axes[1, 0], weight_vector)
        # 4. 加权距离与原始距离对比
        self._plot_distance_comparison(axes[1, 1], target_point, local_points, local_labels, 
                                     weight_vector, reducer, reducer_name)
        plt.suptitle(f'weighted KNN decision boundary analysis (k={k_per_class}/category, method={reducer_name})', fontsize=16)
        plt.tight_layout()
        return fig
    def _plot_weighted_scatter(self, ax, points, labels, test_point, weighted_distances, reducer_name):
        """绘制加权距离下的散点图，点大小反映加权距离"""
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        # 为每个类别绘制点，点大小与加权距离成反比
        for i, label in enumerate(unique_labels):
            if label == -1: # 跳过目标点
                continue
            mask = (labels == label)
            class_points = points[mask]
            class_distances = [weighted_distances[j] for j in range(len(labels)) if mask[j]]
            # 距离越小，点越大（越近的点越突出）
            sizes = [100 / (d + 0.1) for d in class_distances] # 避免除零
            sizes = [min(max(s, 20), 200) for s in sizes] # 限制大小范围
            ax.scatter(class_points[:, 0], class_points[:, 1], 
                      c=[colors[i]], label=f'Class {label}',
                      alpha=0.7, s=sizes, edgecolors='black', linewidth=0.5)
        # 绘制测试点
        ax.scatter(test_point[0], test_point[1], c='red', marker='*',
                  s=300, label='Test Point', edgecolors='black', linewidth=2, zorder=10)
        ax.set_xlabel(f'{reducer_name} Component 1')
        ax.set_ylabel(f'{reducer_name} Component 2')
        ax.set_title('weighted distances of local region\n(size∝1/distance)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    def _plot_weighted_decision_surface(self, ax, local_points, local_labels, target_point, 
                                      weight_vector, reducer, reducer_name):
        """绘制基于加权距离的决策边界"""
        if len(local_points) < 10:
            ax.text(0.5, 0.5, '样本点不足\n无法绘制决策边界', 
                   ha='center', va='center')
            ax.set_title('决策边界 (样本不足)')
            return
        try:
            # 创建网格
            reduced_points = reducer.transform(local_points)
            x_min, x_max = reduced_points[:, 0].min() - 0.5, reduced_points[:, 0].max() + 0.5
            y_min, y_max = reduced_points[:, 1].min() - 0.5, reduced_points[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                               np.linspace(y_min, y_max, 50))
            # 将网格点逆变换到原始空间（近似）
            if hasattr(reducer, 'inverse_transform'):
                grid_points_original = reducer.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
            else:
                # 对于没有逆变换的降维方法，使用最近邻近似
                grid_points_original = self._approximate_inverse_transform(reducer, np.c_[xx.ravel(), yy.ravel()])
            # 预测网格点的类别（基于加权距离）
            Z = np.zeros(xx.ravel().shape)
            for i, grid_point in enumerate(grid_points_original):
                # 计算到所有局部点的加权距离
                distances = []
                for j, train_point in enumerate(local_points):
                    dist = self._calculate_weighted_distance(grid_point, train_point, weight_vector)
                    distances.append((dist, local_labels[j]))
                # 找到最近的几个点，投票决定类别
                distances.sort(key=lambda x: x[0])
                k = min(5, len(distances))
                nearest_classes = [cls for dist, cls in distances[:k]]
                if nearest_classes:
                    Z[i] = Counter(nearest_classes).most_common(1)[0][0]
                else:
                    Z[i] = 0
            Z = Z.reshape(xx.shape)
            # 绘制决策边界
            from matplotlib.colors import ListedColormap
            cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFAAFF'])
            ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
            # 绘制训练点
            unique_labels = np.unique(local_labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            for i, label in enumerate(unique_labels):
                mask = (local_labels == label)
                class_points = reduced_points[mask]
                ax.scatter(class_points[:, 0], class_points[:, 1], 
                          c=[colors[i]], label=f'Class {label}', alpha=0.7, s=30)
            # 绘制测试点
            test_point_reduced = reducer.transform(target_point.reshape(1, -1))
            ax.scatter(test_point_reduced[0, 0], test_point_reduced[0, 1], 
                      c='red', marker='*', s=200, label='Test Point')
            ax.set_xlabel(f'{reducer_name} Component 1')
            ax.set_ylabel(f'{reducer_name} Component 2')
            ax.set_title('weighted decision boundary')
            ax.legend(loc='best')
        except Exception as e:
            ax.text(0.5, 0.5, f'绘制边界错误\n{str(e)[:50]}...', 
                   ha='center', va='center')
            ax.set_title('决策边界 (错误)')
    def _plot_feature_weights(self, ax, weight_vector):
        """绘制属性权重条形图"""
        if self.feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(len(weight_vector))]
        # 排序权重和特征名称
        indices = np.argsort(weight_vector)[::-1] # 降序排列
        sorted_weights = weight_vector[indices]
        sorted_names = [self.feature_names[i] for i in indices]
        colors = plt.cm.viridis(np.linspace(0, 1, len(weight_vector)))
        bars = ax.bar(range(len(sorted_weights)), sorted_weights, color=colors)
        ax.set_xlabel('feature')
        ax.set_ylabel('weight')
        ax.set_title('feature weight distribution')
        ax.set_xticks(range(len(sorted_names)))
        ax.set_xticklabels(sorted_names, rotation=45, ha='right')
        # 添加数值标签
        for bar, weight in zip(bars, sorted_weights):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{weight:.3f}', ha='center', va='bottom', fontsize=8)
    def _plot_distance_comparison(self, ax, target_point, local_points, local_labels, 
                                weight_vector, reducer, reducer_name):
        """绘制加权距离与原始距离的对比"""
        if len(local_points) == 0:
            ax.text(0.5, 0.5, '无数据可比较', ha='center', va='center')
            ax.set_title('距离对比 (无数据)')
            return
        # 计算两种距离
        original_distances = []
        weighted_distances = []
        for point in local_points:
            # 原始欧氏距离
            orig_dist = np.linalg.norm(point - target_point)
            original_distances.append(orig_dist)
            # 加权距离
            weighted_dist = self._calculate_weighted_distance(point, target_point, weight_vector)
            weighted_distances.append(weighted_dist)
        # 绘制散点对比图
        unique_labels = np.unique(local_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = (local_labels == label)
            label_original_dists = [original_distances[j] for j in range(len(local_labels)) if mask[j]]
            label_weighted_dists = [weighted_distances[j] for j in range(len(local_labels)) if mask[j]]
            ax.scatter(label_original_dists, label_weighted_dists, 
                      c=[colors[i]], label=f'Class {label}', alpha=0.6, s=50)
        # 添加对角线参考线
        max_dist = max(max(original_distances), max(weighted_distances))
        ax.plot([0, max_dist], [0, max_dist], 'k--', alpha=0.5, label='y=x')
        ax.set_xlabel('original distances')
        ax.set_ylabel('weighted distances')
        ax.set_title('comparison of distance methods')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    def _approximate_inverse_transform(self, reducer, points):
        """近似逆变换（用于t-SNE等没有逆变换的降维方法）"""
        from sklearn.neighbors import NearestNeighbors
        if hasattr(reducer, 'embedding_'):
            X_original = reducer._X
            X_reduced = reducer.embedding_
        else:
            X_original = self.X
            X_reduced = reducer.transform(self.X)
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_reduced)
        distances, indices = nbrs.kneighbors(points)
        return X_original[indices.flatten()]
    def plot_simple_weighted_view(self, target_point, weight_vector, k_neighbors=50, 
                                method='pca', figsize=(12, 5)):
        """
        简化的加权KNN视图
        """
        # 找到基于加权距离的k个最近邻
        distances = []
        for i in range(len(self.X)):
            dist = self._calculate_weighted_distance(self.X[i], target_point, weight_vector)
            distances.append((i, dist, self.y[i]))
        distances.sort(key=lambda x: x[1])
        nearest_indices = [idx for idx, dist, cls in distances[:k_neighbors]]
        nearest_points = self.X[nearest_indices]
        nearest_labels = [cls for idx, dist, cls in distances[:k_neighbors]]
        nearest_distances = [dist for idx, dist, cls in distances[:k_neighbors]]
        # 添加目标点
        all_points = np.vstack([nearest_points, target_point.reshape(1, -1)])
        all_labels = np.append(nearest_labels, -1)
        # 降维
        if method == 'pca':
            reducer = PCA(n_components=2)
            reduced_points = reducer.fit_transform(all_points)
        else:
            reducer = TSNE(n_components=2, random_state=42)
            reduced_points = reducer.fit_transform(all_points)
        test_point_reduced = reduced_points[-1]
        neighbor_points_reduced = reduced_points[:-1]
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        # 左图：加权最近邻
        unique_labels = np.unique(nearest_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = (nearest_labels == label)
            class_points = neighbor_points_reduced[mask]
            class_distances = [nearest_distances[j] for j in range(len(nearest_labels)) if mask[j]]
            sizes = [80 / (d + 0.1) for d in class_distances]
            sizes = [min(max(s, 20), 150) for s in sizes]
            ax1.scatter(class_points[:, 0], class_points[:, 1], 
                       c=[colors[i]], label=f'Class {label}', alpha=0.7, s=sizes)
        ax1.scatter(test_point_reduced[0], test_point_reduced[1], c='red', marker='*',
                   s=200, label='Test Point')
        ax1.set_xlabel('Component 1')
        ax1.set_ylabel('Component 2')
        ax1.set_title(f'Top {k_neighbors} weighted nearest neighbors')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # 右图：距离分布
        for i, label in enumerate(unique_labels):
            mask = (nearest_labels == label)
            label_dists = [nearest_distances[j] for j in range(len(nearest_labels)) if mask[j]]
            ax2.hist(label_dists, alpha=0.6, label=f'Class {label}', bins=10)
        ax2.axvline(x=np.mean(nearest_distances), color='red', linestyle='--', 
                   label=f'average distance: {np.mean(nearest_distances):.3f}')
        ax2.set_xlabel('weighted distances')
        ax2.set_ylabel('frequency')
        ax2.set_title('weighted distances distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
class KNNVisualizer:
    """加权KNN分类器可视化工具"""
    def __init__(self, classifier, feature_names=None):
        self.classifier = classifier
        self.feature_names = feature_names
        self.set_style()
    def set_style(self):
        """设置绘图风格"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
    def plot_feature_importance(self, comprehensive_weights, figsize=(12, 6)):
        """绘制特征重要性（权重）热力图"""
        fig, ax = plt.subplots(figsize=figsize)
        # 创建权重矩阵
        classes = list(comprehensive_weights.keys())
        weights_matrix = np.array([comprehensive_weights[cls] for cls in classes])
        # 绘制热力图
        im = ax.imshow(weights_matrix, cmap='YlOrRd', aspect='auto')
        # 设置坐标轴
        ax.set_xticks(range(len(self.feature_names)))
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels([f'Class {cls}' for cls in classes])
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Feature Weight', rotation=270, labelpad=15)
        # 添加数值标注
        for i in range(len(classes)):
            for j in range(len(self.feature_names)):
                text = ax.text(j, i, f'{weights_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=8)
        ax.set_title("Feature Weights for Different Classes")
        plt.tight_layout()
        return fig
    def plot_distance_distribution(self, knn_results, figsize=(10, 6)):
        """绘制距离分布直方图"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        # 提取所有距离数据
        all_distances = []
        for cls, result in knn_results.items():
            distances = result['distances']
            all_distances.extend(distances)
            axes[0].hist(distances, alpha=0.6, label=f'Class {cls}', bins=20)
        axes[0].set_xlabel('Distance')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distance Distribution by Class')
        axes[0].legend()
        # 整体距离分布
        axes[1].hist(all_distances, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1].set_xlabel('Distance')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Overall Distance Distribution')
        # 添加统计信息
        mean_dist = np.mean(all_distances)
        std_dist = np.std(all_distances)
        axes[1].axvline(mean_dist, color='red', linestyle='--', 
                       label=f'Mean: {mean_dist:.2f}')
        axes[1].axvline(mean_dist + std_dist, color='orange', linestyle=':',
                       label=f'±1 STD')
        axes[1].axvline(mean_dist - std_dist, color='orange', linestyle=':')
        axes[1].legend()
        plt.tight_layout()
        return fig
    def plot_voting_results(self, weighted_votes, true_class=None, figsize=(10, 6)):
        """绘制投票结果条形图"""
        fig, ax = plt.subplots(figsize=figsize)
        classes = list(weighted_votes.keys())
        votes = list(weighted_votes.values())
        colors = ['lightblue' if cls != true_class else 'lightcoral' 
                 for cls in classes]
        bars = ax.bar(range(len(classes)), votes, color=colors, alpha=0.7, 
                     edgecolor='black')
        ax.set_xlabel('Classes')
        ax.set_ylabel('Weighted Votes')
        ax.set_title('Weighted Voting Results by Class')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([f'Class {cls}' for cls in classes])
        # 添加数值标签
        for bar, vote in zip(bars, votes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{vote:.4f}', ha='center', va='bottom')
        if true_class is not None:
            # 标记真实类别
            true_idx = classes.index(true_class)
            bars[true_idx].set_color('lightgreen')
            ax.text(true_idx, votes[true_idx] + 0.01, 'True Class', 
                   ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        return fig
    def plot_neighbor_analysis(self, knn_results, target_point_idx, figsize=(12, 8)):
        """绘制近邻分析图"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        for idx, (cls, result) in enumerate(knn_results.items()):
            if idx >= 4: # 最多显示4个类别
                break
            neighbors = result['neighbors']
            distances = result['distances']
            neighbor_classes = result['neighbor_classes']
            # 距离排序图
            axes[idx].plot(range(1, len(distances) + 1), distances, 'o-', 
                          markersize=6, linewidth=2)
            axes[idx].set_xlabel('Neighbor Rank')
            axes[idx].set_ylabel('Distance')
            axes[idx].set_title(f'Class {cls}: Distance vs Rank')
            axes[idx].grid(True, alpha=0.3)
            # 添加同类近邻标记
            same_class_indices = [i for i, c in enumerate(neighbor_classes) if c == cls]
            if same_class_indices:
                axes[idx].scatter([i+1 for i in same_class_indices], 
                                [distances[i] for i in same_class_indices],
                                color='red', s=50, zorder=5, label='Same Class')
                axes[idx].legend()
        plt.tight_layout()
        return fig
    def plot_confusion_matrix_comparison(self, y_true, y_pred, class_names=None, 
                                       figsize=(10, 8)):
        """绘制混淆矩阵比较图"""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]
        # 设置坐标轴
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               title='Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # 添加数值标注
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return fig
    def create_comprehensive_report(self, X, y, target_point, comprehensive_weights,
                                  local_region_classes, knn_results, weighted_votes,
                                  true_class=None, predicted_class=None):
        """生成综合可视化报告"""
        print("生成综合可视化报告...")
        # 创建大图
        fig = plt.figure(figsize=(20, 16))
        # 1. 特征重要性热力图
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        weights_matrix = np.array([comprehensive_weights[cls] 
                                 for cls in comprehensive_weights.keys()])
        im = ax1.imshow(weights_matrix, cmap='YlOrRd', aspect='auto')
        ax1.set_title("Feature Weights Heatmap")
        ax1.set_xticks(range(len(self.feature_names)))
        ax1.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax1.set_yticks(range(len(comprehensive_weights.keys())))
        ax1.set_yticklabels([f'Class {cls}' for cls in comprehensive_weights.keys()])
        plt.colorbar(im, ax=ax1)
        # 2. 投票结果
        ax2 = plt.subplot2grid((3, 3), (0, 2))
        classes = list(weighted_votes.keys())
        votes = list(weighted_votes.values())
        colors = ['lightgreen' if cls == predicted_class else 'lightblue' 
                 for cls in classes]
        bars = ax2.bar(classes, votes, color=colors, alpha=0.7)
        ax2.set_title("Weighted Voting Results")
        ax2.set_ylabel("Vote Score")
        # 3. 距离分布
        ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=3)
        all_distances = []
        for cls, result in knn_results.items():
            distances = result['distances']
            all_distances.extend(distances)
            ax3.hist(distances, alpha=0.6, label=f'Class {cls}', bins=15)
        ax3.set_title("Distance Distribution by Class")
        ax3.set_xlabel("Distance")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        # 4. 近邻分析
        ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        for cls, result in knn_results.items():
            distances = result['distances']
            ax4.plot(range(1, len(distances) + 1), distances, 'o-', 
                    label=f'Class {cls}', markersize=4)
        ax4.set_title("Neighbor Distance Analysis")
        ax4.set_xlabel("Neighbor Rank")
        ax4.set_ylabel("Distance")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
# ==================== 主函数 ====================
def main():
    """主函数"""
    # 文件路径
    file_path = 'D:/Downloads/archive/transactions.csv'
    try:
        # 1. 加载和预处理数据
        print("=== 数据加载和预处理 ===")
        df = pd.read_csv(file_path)
        # 添加行号列（CSV文件中的实际行号，从2开始，因为第一行是标题）
        df['csv_row_number'] = range(2, len(df) + 2)  # 从2开始，因为第一行是标题
        df = df.dropna()
        # df['type'] = df['type'].map({'red': 1, 'white': 2, 'Red': 1, 'White': 2}).fillna(0)
        # df['class'] = df['class'].map({'abnormal': 0, 'normal': 1}).fillna(0)
        # df['protocol_type'] = df['protocol_type'].map({'tcp': 0, 'udp': 1,'icmp':2}).fillna(0)
        # df['service'] = pd.factorize(df['service'])[0]
        # df['flag'] = pd.factorize(df['flag'])[0]
        # df['GENDER'] = pd.factorize(df['GENDER'])[0]
        # df['CAR'] = pd.factorize(df['CAR'])[0]
        # df['REALITY'] = pd.factorize(df['REALITY'])[0]
        # df['INCOME_TYPE'] = pd.factorize(df['INCOME_TYPE'])[0]
        # df['EDUCATION_TYPE'] = pd.factorize(df['EDUCATION_TYPE'])[0]
        # df['FAMILY_TYPE'] = pd.factorize(df['FAMILY_TYPE'])[0]
        # df['HOUSE_TYPE'] = pd.factorize(df['HOUSE_TYPE'])[0]
        # df['species'] = pd.factorize(df['species'])[0]
        df['merchant_category'] = pd.factorize(df['merchant_category'])[0]
        df['country'] = pd.factorize(df['country'])[0]
        df['bin_country'] = pd.factorize(df['bin_country'])[0]
        df['channel'] = pd.factorize(df['channel'])[0]
        # df['Attrition'] = pd.factorize(df['Attrition'])[0]
        # df['BusinessTravel'] = pd.factorize(df['BusinessTravel'])[0]
        # df['Department'] = pd.factorize(df['Department'])[0]
        # df['EducationField'] = pd.factorize(df['EducationField'])[0]
        # df['Gender'] = pd.factorize(df['Gender'])[0]
        # df['JobRole'] = pd.factorize(df['JobRole'])[0]
        # df['MaritalStatus'] = pd.factorize(df['MaritalStatus'])[0]
        # df['Over18'] = pd.factorize(df['Over18'])[0]
        # df['gender'] = pd.factorize(df['gender'])[0]
        # df['grade_level'] = pd.factorize(df['grade_level'])[0]
        # df['ai_tools_used'] = pd.factorize(df['ai_tools_used'])[0]
        # df['ai_usage_purpose'] = pd.factorize(df['ai_usage_purpose'])[0]

        
        # 分离特征、标签和行号
        # 注意：行号不作为特征使用，只用于追踪
        row_indices = df['csv_row_number'].values
        # X = df.drop(['quality', 'csv_row_number'], axis=1).values
        # y = df['quality'].values
        # X = df.drop(['species', 'csv_row_number'], axis=1).values
        # y = df['species'].values
        # X = df.drop(['class', 'csv_row_number'], axis=1).values
        # y = df['class'].values
        # X = df.drop(['row','TARGET', 'csv_row_number'], axis=1).values
        # y = df['TARGET'].values
        X = df.drop(['is_fraud','transaction_time','transaction_id', 'csv_row_number'], axis=1).values
        y = df['is_fraud'].values
        # X = df.drop(['EmployeeID','PerformanceRating','csv_row_number'], axis=1).values
        # y = df['PerformanceRating'].values
        # X = df.drop(['Diabetes_012','csv_row_number'], axis=1).values
        # y = df['Diabetes_012'].values
        # X = df.drop(['Class','csv_row_number'], axis=1).values
        # y = df['Class'].values
        # X = df.drop(['student_id','performance_category','csv_row_number'], axis=1).values
        # y = df['performance_category'].values
        # X = df.drop(['transaction_id','is_fraud','csv_row_number'], axis=1).values
        # y = df['is_fraud'].values
        print(f"数据形状: X{X.shape}, y{y.shape}")
        print("类别分布:", Counter(y))
        # feature_names = df.drop(['quality', 'csv_row_number'], axis=1).columns.tolist()
        # feature_names = df.drop(['species', 'csv_row_number'], axis=1).columns.tolist()
        # feature_names = df.drop(['TARGET','row', 'csv_row_number'], axis=1).columns.tolist()
        # feature_names = df.drop(['class', 'csv_row_number'], axis=1).columns.tolist()
        feature_names = df.drop(['is_fraud','transaction_time', 'csv_row_number'], axis=1).columns.tolist()
        # feature_names = df.drop(['EmployeeID','PerformanceRating', 'csv_row_number'], axis=1).columns.tolist()
        # feature_names = df.drop(['Diabetes_012', 'csv_row_number'], axis=1).columns.tolist()
        # feature_names = df.drop(['performance_category','student_id', 'csv_row_number'], axis=1).columns.tolist()
        # feature_names = df.drop(['is_fraud','transaction_id', 'csv_row_number'], axis=1).columns.tolist()
         
        print("数据属性:", feature_names)

        # 2. 划分训练测试集
        """划分训练测试集，同时保留行号信息"""
        # 使用相同的随机种子确保划分一致
        X_train, X_test, y_train, y_test, row_indices_train, row_indices_test = train_test_split(
            X, y, row_indices, 
            test_size=0.0002, 
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
        # print("逆协方差打印",inv_cov_matrix)
        # 3. 开始测试：选择测试样本并统计正确预测数
        correct_count = 0
        total_count = len(X_test)
        test_result={}
        test_result['details'] = {}
        unfound_count=0
        corrector = BinaryClassWeightCorrector()
        corrector.fit(X_train,y_train)
        # local_cov_corrector = ManifoldWeightCorrector()
        # local_corrector = BinaryClassWeightCorrector()
        weight_calculator = GlobalWeightMatrix()
        region_builder = SmoothLocalRegionBuilder( k_region=15,
                                                    region_size=20,
                                                    # winequalityN 数据集最佳参数 k_region=15 region_size=20
                                                    alpha=1.0, # 贝叶斯先验
                                                    beta=1.0, # 贝叶斯先验
                                                    confidence_level=0.95, # 威尔逊置信水平
                                                    sigma=0.5 # 高斯核带宽
                                                )
        region_builder.fit(X_train,y_train,row_indices_train)
        # 使用加权KNN进行分类
        knn_classifier = WeightedKNNClassifier(k=7, inv_cov_matrix= inv_cov_matrix)
        knn_classifier.fit(X_train, y_train, row_indices_train, feature_names)
        # 初始化可视化工具
        local_visualizer = WeightedKNNVisualizer(classifier=knn_classifier, feature_names=feature_names)
        local_visualizer.fit(X_train, y_train)
        knn_visualizer = KNNVisualizer(classifier=knn_classifier, feature_names=feature_names)
        
        for target_idx in range(len(X_test)):
            temp_result= {}
            target_point = X_test[target_idx]
            target_point_normalized = X_test_normalized[target_idx]
            target_class = y_test[target_idx]
            target_point_raw_data = X_train[target_idx]
            target_row = row_indices_test[target_idx]
            # print(f"\n=== 对测试样本进行局部区域搜索 ===")
            # print(f"目标样本真实类别: {target_class}")
            # 4. 使用基于双向排名的类别代表性搜索最佳局部区域
            best_region = region_builder.build_local_region(target_point)
            # 5. 分析结果
            # print(f"\n=== 最终搜索结果 ===")
            # print(f"最佳适应度: {best_fitness:.4f}")
            # print(f"局部区域包含样本数: {len(best_region)}")
            # print(f"局部区域内类别分布: {dict(best_class_dist)}")
            # target_ratio = best_class_dist.get(target_class, 0) / len(best_region)
            # print(f"目标类别在局部区域中的比例: {target_ratio:.4f}")
            # 6. 分析局部区域是否包含目标类别
            # if target_class in best_class_dist:
            #     print("✅ 成功找到包含目标类别的局部区域!")
            if best_region is not None:
                print(f"\n" + "="*60)
                print("局部区域搜索完成!")
                print(f"找到了包含 {len(best_region)} 个样本的局部区域")
                print(f"找到了best_region的局部区域 {row_indices_train[best_region]} ")
                print(f"目标类别 {target_class} 在区域中的比例: {Counter(y_train[best_region]).get(target_class, 0)/len(best_region):.4f}")
                print("="*60)
                
            # 6. 提取局部区域数据
            X_region_row_indices_train = row_indices_train[best_region]
            X_region_raw_data = X_train[best_region]
            X_region = X_train[best_region]
            y_region = y_train[best_region]
            weight_calculator.fit(X_region,y_region)
            
            
            # 7. 分析局部区域内的属性特性(质心法)
            # analyzer = LocalRegionAnalyzer()
            # analyzer = CorrectedLocalRegionAnalyzer()
            # analysis_results = analyzer.analyze_local_region(X_region, y_region, feature_names)
            # comprehensive_weights = analysis_results['comprehensive_weights']
            # # 8. 计算全局信息增益
            # global_mi = analyzer.calculate_global_mutual_info(X_region, y_region, feature_names)
            # analysis_results['global_mutual_info'] = global_mi
            # # 9. 计算综合权重
            # comprehensive_weights = analyzer.calculate_comprehensive_weights(
            #     analysis_results, feature_names
            # )
            
            comprehensive_weights = weight_calculator.get_class_weight_dict()
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
            
            
            # 获取局部区域中包含的类别
            local_region_classes = list(set(y_region))
            # 进行预测
            predicted_class, consistency, knn_results,weighted_votes = knn_classifier.predict_with_voting(
                target_point, comprehensive_weights, local_region_classes
            )
            print(f"目标类别：{target_class}")
            print(f"预测结果 for {target_row}: {'是' if target_class == predicted_class else '否'}")
            # # 11. 评估一致性
            # true_class_consistency, confidence_scores = knn_classifier.evaluate_consistency(knn_results, target_class)
            # 0. 绘制分类边界
            # weight_vector = comprehensive_weights[target_class]
            # fig1 = local_visualizer.plot_weighted_decision_boundary(
            #                                         target_point=target_point,
            #                                         weight_vector=weight_vector,
            #                                         k_per_class=20,
            #                                         method='pca'
            #                                     )
            # fig2 = local_visualizer.plot_simple_weighted_view(
            #                                         target_point=target_point,
            #                                         weight_vector=weight_vector,
            #                                         k_neighbors=50
            #                                     )
            # # 1. 绘制特征重要性热力图
            # fig1 = visualizer.plot_feature_importance(comprehensive_weights)
            # # plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            # # 2. 绘制投票结果
            # fig2 = visualizer.plot_voting_results(weighted_votes, true_class=target_class)
            # # plt.savefig('voting_results.png', dpi=300, bbox_inches='tight')
            # # 3. 绘制距离分布
            # fig3 = visualizer.plot_distance_distribution(knn_results)
            # # plt.savefig('distance_distribution.png', dpi=300, bbox_inches='tight')
            # # 4. 绘制近邻分析
            # fig4 = visualizer.plot_neighbor_analysis(knn_results, target_point_idx=0)
            # plt.savefig('neighbor_analysis.png', dpi=300, bbox_inches='tight')
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
            # temp_result['region_class_dist'] = best_class_dist
            temp_result['best_region'] = X_region_row_indices_train
            temp_result['X_region_raw_data'] = X_region_raw_data
            # temp_result['analysis_results'] = analysis_results
            # temp_result['comprehensive_weights'] = comprehensive_weights
            # temp_result['knn_results'] = knn_results
            # temp_result['consistency'] = weighted_votes
            # temp_result['predicted_class'] = predicted_class
            # temp_result['true_class_consistency'] = true_class_consistency
            # temp_result['confidence_scores'] = confidence_scores
            # temp_result['correctOrNot'] = True if predicted_class == target_class else False
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
    print(f"运行完毕：untitled20 使用马氏距离")
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
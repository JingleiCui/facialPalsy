┌────────────────────────────────────────────────────────────┐
│           H-GFA Net 完整数据处理与训练流程                  │
└────────────────────────────────────────────────────────────┘

[数据库] → [预处理] → [特征提取] → [模型训练]

阶段1: 预处理与峰值帧检测
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: 11个动作视频 (每个动作一个视频片段)

对每个视频:
  ↓
[1] 逐帧提取478关键点 (MediaPipe FaceLandmarker)
  → 得到: landmarks_sequence (T, 478, 3)
  
  ↓
[2] 计算一级几何指标 (逐帧)
  → 眼睛开合度、嘴角距离、眉眼距、鼻唇沟深度...
  → 得到: primary_metrics_sequence (T, N_metrics)
  
  ↓
[3] 峰值帧检测 (Peak Frame Detection)
  ⚠️ 关键: 不同动作使用不同的检测逻辑
  
  例如:
  • CloseEyeHardly → 检测眼睛开合度最小的帧
  • RaiseEyebrow → 检测眉眼距最大的帧
  • Smile → 检测嘴角上扬最大的帧
  • NeutralFace → 取中间段稳定帧
  
  → 得到: peak_frame_idx
  
  ↓
[4] 保存结果
  • 峰值帧图像: peak_frame.jpg
  • 关键点序列: landmarks_sequence.npy
  • 一级指标序列: metrics_sequence.npy
  • 峰值帧索引: peak_frame_idx


阶段2: 几何特征提取
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: 关键点序列 + 峰值帧索引

[1] 静态几何特征 (Static Geometric Features)
  • 来源: 峰值帧那一刻的关键点
  • 计算:
    - 一级指标: 眼睛开合度、嘴角宽度、眉眼距等 (基础测量)
    - 二级指标: 左右对称性、相对位置比例等 (比较/对称性)
  
  → 得到: static_features (32维)

[2] 动态几何特征 (Dynamic Geometric Features)
  • 来源: 整个视频序列的关键点变化
  • 计算:
    - 运动范围: max - min
    - 运动速度: 平均速度、最大速度
    - 运动加速度
    - 运动平滑度 (jerk)
  
  → 得到: dynamic_features (16维)

⚠️ 您的理解需要修正的地方:
  静态特征 ≠ 仅峰值帧
  动态特征 ≠ 仅序列
  
  正确的理解:
  • 静态特征: 描述峰值帧的空间结构
  • 动态特征: 描述整个动作的时序变化
  • 两者是互补的，不是替代关系


阶段3: 视觉特征提取
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: 峰值帧图像

[1] MobileNetV3 提取视觉特征
  peak_frame (224, 224, 3) 
    → MobileNetV3 (pretrained)
    → visual_features (1280维)


阶段4: 特征融合 (多阶段)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[正确架构 - 参考H-GFA Net设计]

  Stage 1: CDCAF (几何特征处理)
  ────────────────────────────────
  static_geo (32维) ──┐
                      ├→ [Multi-Head Attention] → geo_refined (48维)
  dynamic_geo (16维) ─┘
  
  目的: 让静态和动态特征互相增强
  

  Stage 2: GQCA (几何引导的视觉特征)
  ────────────────────────────────
  geo_refined (48维) ──→ [Query]
                          ↓
  visual_features (1280维) → [7×7 spatial map]
                          ↓
                      [Cross Attention]
                          ↓
                   visual_guided (512维)
  
  目的: 用几何特征引导视觉特征关注重要区域
  

  Stage 3: MFA (多模态融合)
  ────────────────────────────────
  三个分支:
  1. geometric_branch: geo_refined → 256维
  2. visual_guided_branch: visual_guided → 256维
  3. visual_global_branch: visual_features (avg pool) → 256维
  
  → [Learnable Weighted Fusion] → action_feature (512维)
  
  权重是自适应学习的，不是固定的！


阶段5: 动作级预测 (Action-Level)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: action_feature (512维)

[MLP]
  action_feature (512)
    → FC(512 → 256) + ReLU + Dropout
    → FC(256 → 5)  # 5个严重程度等级
    → severity_score (1-5)

损失: CrossEntropyLoss
监督: 医生标注的单个动作评分


阶段6: 检查级预测 (Examination-Level) ⚠️ 关键修正
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: 11个动作特征 + 11个动作评分

⚠️ 您的理解需要修正:

[错误理解]
  11个动作特征 → 聚合 → 多任务学习

[正确架构]
  
  [11个动作特征序列]
    action_features (11, 512)
    
    ↓
  
  [Temporal Aggregation]
  
  Attention Pooling
    action_features (11, 512)
      → Attention(Q=learnable, K=V=actions)
      → exam_feature (512)
  
    ↓
  
  [多任务预测头]
  
  exam_feature (512) ─┬→ [MLP] → is_palsy (0/1)
                      │
                      ├→ [MLP] → palsy_side (0/1/2)
                      │
                      ├→ [MLP] → hb_grade (1-6)
                      │
                      └→ [MLP] → sunnybrook_score (0-100, regression)

损失: 
  L_total = λ₁·L_palsy + λ₂·L_side + λ₃·L_hb + λ₄·L_sunnybrook
  
  其中权重 λ 可以:
  1. 使用不确定性加权 (Uncertainty Weighting) 自动学习。

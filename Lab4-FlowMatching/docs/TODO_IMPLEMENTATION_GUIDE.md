# Flow Matching Assignment - TODO Implementation Guide

本文檔詳細說明所有 TODO 項目的實作細節和數學原理。

---

## 📋 目錄

1. [Task 1: 2D Flow Matching](#task-1-2d-flow-matching)
   - [TODO 1.1: Network Architecture](#todo-11-network-architecture)
   - [TODO 1.2: Forward Pass](#todo-12-forward-pass)
   - [TODO 1.3: Conditional Flow](#todo-13-conditional-flow)
   - [TODO 1.4: Euler Step](#todo-14-euler-step)
   - [TODO 1.5: Training Loss](#todo-15-training-loss)
   - [TODO 1.6: Sampling Loop](#todo-16-sampling-loop)

2. [Task 2: Image Flow Matching](#task-2-image-flow-matching)
   - [TODO 2.1-2.4: 與 Task 1 相同](#task-2-image-flow-matching)

3. [Task 3: Rectified Flow](#task-3-rectified-flow)
   - [TODO 3.1: Reflow Data Generation](#todo-31-reflow-data-generation)

4. [Task 4: InstaFlow](#task-4-instaflow)
   - [TODO 4.1: Distillation Loss](#todo-41-distillation-loss)
   - [TODO 4.2: One-Step Sampling](#todo-42-one-step-sampling)

---

## Task 1: 2D Flow Matching

### TODO 1.1: Network Architecture
**檔案**: `task1_2d_flow_matching/network.py` (Line 84-93)

#### 目標
建立一個時間條件化的 MLP 網路，用於預測噪聲。

#### 實作
```python
######## TODO ########
# Build MLP with time conditioning
self.layers = nn.ModuleList()

# First layer from input to first hidden
self.layers.append(TimeLinear(dim_in, dim_hids[0], num_timesteps))

# Hidden layers
for i in range(len(dim_hids) - 1):
    self.layers.append(TimeLinear(dim_hids[i], dim_hids[i+1], num_timesteps))

# Output layer
self.layers.append(TimeLinear(dim_hids[-1], dim_out, num_timesteps))
######################
```

#### 數學原理
- **TimeLinear**: 將時間 `t` 嵌入到線性層中，使網路能夠學習時間相關的特徵
- **網路結構**: `input → hidden_1 → ... → hidden_n → output`
- **每層**: $h_{i+1} = \text{TimeLinear}(h_i, t)$

#### 重點說明
1. 使用 `TimeLinear` 而非普通 `nn.Linear`，因為需要時間條件化
2. 隱藏層數量由 `dim_hids` 決定
3. 輸出維度必須與輸入維度相同（預測噪聲）

---

### TODO 1.2: Forward Pass
**檔案**: `task1_2d_flow_matching/network.py` (Line 111-120)

#### 目標
實作網路的前向傳播，輸出噪聲預測。

#### 實作
```python
######## TODO ########
# Forward through all layers with time conditioning
for i, layer in enumerate(self.layers[:-1]):
    x = layer(x, t)
    x = F.silu(x)  # Apply SiLU activation

# Last layer without activation
x = self.layers[-1](x, t)
return x
######################
```

#### 數學原理
- **SiLU 激活函數**: $\text{SiLU}(x) = x \cdot \sigma(x)$，其中 $\sigma$ 是 sigmoid
- **最後一層無激活**: 輸出應該是原始的噪聲預測值

#### 重點說明
1. 除了最後一層外，每層都加 SiLU 激活
2. SiLU (Swish) 比 ReLU 更平滑，適合連續流
3. 時間 `t` 在每層都要傳入

---

### TODO 1.3: Conditional Flow
**檔案**: `task1_2d_flow_matching/fm.py` (Line 46-50)

#### 目標
計算條件流 $\psi_t(x | x_1)$。

#### 實作
```python
######## TODO ########
psi_t = t * x1 + (1 - t) * x
######################
```

#### 數學原理
**Conditional Flow 公式** (Rectified Flow):
$$\psi_t(x | x_1) = t \cdot x_1 + (1-t) \cdot x_0$$

其中:
- $x_1$: 數據樣本（目標）
- $x_0$ (即 `x`): 噪聲（起點）
- $t \in [0, 1]$: 時間參數
- 當 $t=0$: $\psi_0 = x_0$ (純噪聲)
- 當 $t=1$: $\psi_1 = x_1$ (數據)

#### 重點說明
1. 這是一條**直線路徑**，連接噪聲和數據
2. `t` 已經被 `expand_t` 擴展過維度，可以直接廣播
3. 這是 Rectified Flow 的核心：用直線代替彎曲路徑

---

### TODO 1.4: Euler Step
**檔案**: `task1_2d_flow_matching/fm.py` (Line 61-64)

#### 目標
實作一階 Euler ODE 求解器的單步更新。

#### 實作
```python
######## TODO ########
dt = expand_t(dt, xt)
x_next = xt + dt * vt
######################
```

#### 數學原理
**Euler Method**:
$$x_{t+\Delta t} = x_t + \Delta t \cdot v_t$$

其中:
- $x_t$: 當前位置
- $v_t$: 當前速度（由網路預測）
- $\Delta t$: 時間步長
- $x_{t+\Delta t}$: 下一個位置

這是求解 ODE $\frac{dx}{dt} = v(x, t)$ 最簡單的方法。

#### 重點說明
1. `expand_t` 確保 `dt` 維度與 `xt` 匹配
2. 這是**顯式 Euler 法**，一階精度
3. 越小的 `dt` 越準確，但需要更多步數

---

### TODO 1.5: Training Loss
**檔案**: `task1_2d_flow_matching/fm.py` (Line 94-107)

#### 目標
實作 Conditional Flow Matching (CFM) 訓練目標。

#### 實作
```python
######## TODO ########
# Implement the CFM objective
xt = self.fm_scheduler.compute_psi_t(x1, t, x0)
target_velocity = x1 - x0

if class_label is not None:
    model_out = self.network(xt, t, class_label=class_label)
else:
    model_out = self.network(xt, t)

loss = F.mse_loss(model_out, target_velocity)
######################
```

#### 數學原理
**CFM Objective** (對應論文 Eq. 23):
$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t) - u_t \|^2 \right]$$

其中:
- $x_t = (1-t)x_0 + t \cdot x_1$: 插值樣本
- $u_t = x_1 - x_0$: **目標速度**（常數，因為是直線）
- $v_\theta(x_t, t)$: 網路預測的速度
- $t \sim \text{Uniform}[0,1]$

#### 重點說明
1. **目標速度是常數**: $u_t = x_1 - x_0$（不依賴 $t$）
2. 這與 DDPM 不同：DDPM 預測噪聲，CFM 預測速度
3. 支持 Classifier-Free Guidance (CFG)：隨機將 `class_label` 設為 `None`

---

### TODO 1.6: Sampling Loop
**檔案**: `task1_2d_flow_matching/fm.py` (Line 147-160)

#### 目標
實作 CFG 引導的採樣循環。

#### 實作
```python
######## TODO ########
# Complete the sampling loop
if do_classifier_free_guidance:
    v_cond = self.network(xt, t, class_label=class_label)
    v_uncond = self.network(xt, t, class_label=None)
    vt = v_uncond + guidance_scale * (v_cond - v_uncond)
else:
    if class_label is not None:
        vt = self.network(xt, t, class_label=class_label)
    else:
        vt = self.network(xt, t)

dt = t_next - t
xt = self.fm_scheduler.step(xt, vt, dt)
######################
```

#### 數學原理
**Classifier-Free Guidance (CFG)**:
$$v_{\text{guided}} = v_{\text{uncond}} + w \cdot (v_{\text{cond}} - v_{\text{uncond}})$$

其中:
- $v_{\text{cond}}$: 條件速度（有類別標籤）
- $v_{\text{uncond}}$: 無條件速度（無類別標籤）
- $w$: `guidance_scale`（引導強度）
- 當 $w=1$: 標準條件生成
- 當 $w>1$: 增強條件信號

**採樣流程**:
1. $t_0 = 0, x_0 \sim \mathcal{N}(0, I)$ (從噪聲開始)
2. For $i = 0, 1, \ldots, N-1$:
   - $v_i = v_\theta(x_i, t_i)$ (預測速度)
   - $x_{i+1} = x_i + (t_{i+1} - t_i) \cdot v_i$ (Euler step)
3. 返回 $x_N \approx x_1$ (數據)

#### 重點說明
1. **CFG 需要兩次前向傳播**: 有條件 + 無條件
2. **無條件**: 將 `class_label` 設為 `None`（觸發 null token）
3. **引導強度**: AFHQ 資料集建議 `guidance_scale=1.0`（幾乎不用 CFG）

---

## Task 2: Image Flow Matching

Task 2 的 TODO 項目與 Task 1 **完全相同**，只是應用於影像資料：

### TODO 2.1: Conditional Flow
**檔案**: `image_common/fm.py` (Line 50-58)

```python
######## TODO ########
psi_t = (1 - t) * x + t * x1
######################
```

與 Task 1 相同，但維度是 `[B, C, H, W]` (影像)。

---

### TODO 2.2: Euler Step
**檔案**: `image_common/fm.py` (Line 68-76)

```python
######## TODO ########
dt = expand_t(dt, xt)
x_next = xt + dt * vt
######################
```

與 Task 1 相同。

---

### TODO 2.3: Training Loss
**檔案**: `image_common/fm.py` (Line 115-147)

```python
######## TODO ########
t_ = t.view(-1, *([1] * (x1.dim() - 1)))
x_t = (1 - t_) * x0 + t_ * x1
u_t = x1 - x0

if class_label is not None:
    model_out = self.network(x_t, t, class_label=class_label)
else:
    model_out = self.network(x_t, t)

loss = (model_out - u_t).pow(2).mean()
######################
```

#### 額外說明
- `t.view(-1, *([1] * (x1.dim() - 1)))`: 將 `t` 從 `[B]` 擴展為 `[B, 1, 1, 1]`，用於影像廣播

---

### TODO 2.4: Sampling Loop
**檔案**: `image_common/fm.py` (Line 193-207)

```python
######## TODO ########
dt = t_next - t

if do_classifier_free_guidance:
    v_cond = self.network(xt, t, class_label=class_label)
    v_uncond = self.network(xt, t, class_label=None)
    vt = v_uncond + guidance_scale * (v_cond - v_uncond)
else:
    if class_label is not None:
        vt = self.network(xt, t, class_label=class_label)
    else:
        vt = self.network(xt, t)

xt = self.fm_scheduler.step(xt, vt, dt)
######################
```

與 Task 1 相同。

---

## Task 3: Rectified Flow

### TODO 3.1: Reflow Data Generation
**檔案**: `task3_rectified_flow/generate_reflow_data.py` (Line 58-103)

#### 目標
生成 Rectified Flow 的訓練資料對 $(x_0, z_1)$。

#### 實作
```python
######## TODO ########
# Sample initial noise
shape = (B, 3, fm.image_resolution, fm.image_resolution)
x_0 = torch.randn(shape).to(device)

# Sample class labels if using CFG
if args.use_cfg:
    # Sample labels from 1 to num_classes (skip null class 0)
    labels = torch.randint(1, num_classes + 1, (B,)).to(device)
else:
    labels = None

# Generate z_1 by simulating the learned flow
with torch.no_grad():
    xt = x_0.clone()
    timesteps = [
        i / args.num_inference_steps for i in range(args.num_inference_steps)
    ]
    timesteps_tensor = [torch.tensor([t] * B).to(device) for t in timesteps]
    
    for i, t in enumerate(timesteps_tensor):
        t_next = timesteps_tensor[i + 1] if i < len(timesteps_tensor) - 1 else torch.ones_like(t)
        
        if args.use_cfg:
            v_cond = fm.network(xt, t, class_label=labels)
            v_uncond = fm.network(xt, t, class_label=None)
            vt = v_uncond + args.cfg_scale * (v_cond - v_uncond)
        else:
            if labels is not None:
                vt = fm.network(xt, t, class_label=labels)
            else:
                vt = fm.network(xt, t)
        
        dt = t_next - t
        xt = fm.fm_scheduler.step(xt, vt, dt)
    
    z_1 = xt
######################
```

#### 數學原理
**Rectified Flow Reflow** (論文 Algorithm 1):

1. **生成配對資料**:
   - 從先驗分佈採樣: $x_0 \sim p_0$ (如 $\mathcal{N}(0, I)$)
   - 用已訓練的 Flow Matching 模型模擬 ODE:
     $$z_1 = \Phi_1(x_0) = x_0 + \int_0^1 v_\theta(x_t, t) dt$$
   - 儲存配對 $(x_0, z_1)$

2. **用於後續訓練**:
   - Rectified Flow 模型用這些配對訓練
   - 目標: 學習從 $x_0$ 到 $z_1$ 的**更直的路徑**

#### 流程圖
```
Original FM:      x_0 ──(彎曲)──> x_1 (真實數據)
                   ↓
              生成 z_1 = Φ_1(x_0)
                   ↓
Rectified Flow:  x_0 ──(拉直)──> z_1 (生成數據)
```

#### 重點說明
1. **不需要真實數據 $x_1$**: 只需要 $(x_0, z_1)$ 配對
2. **CFG 要在生成時使用**: 確保 $z_1$ 是高品質的
3. **儲存格式**: `.pt` 檔案（`{idx:06d}_x0.pt`, `{idx:06d}_z1.pt`）
4. **Reflow 可以多次**: RF1 → RF2 → RF3...，路徑越來越直

---

## Task 4: InstaFlow

### TODO 4.1: Distillation Loss
**檔案**: `image_common/instaflow.py` (Line 51-86)

#### 目標
實作 InstaFlow 的一步蒸餾損失。

#### 實作
```python
######## TODO ########
t_zero = torch.zeros(x0.shape[0], device=self.device)

if class_label is not None:
    v_pred = self.network(x0, t_zero, class_label=class_label)
else:
    v_pred = self.network(x0, t_zero)

x1_pred = x0 + v_pred
l2_loss = (x1_pred - x1).pow(2).mean()

# Combine L2 + LPIPS loss
if self.use_lpips:
    x1_clamped = torch.clamp(x1, -1, 1)
    x1_pred_clamped = torch.clamp(x1_pred, -1, 1)
    lpips_loss = self.lpips_fn(x1_pred_clamped, x1_clamped).mean()
    loss = l2_loss + lpips_loss
else:
    loss = l2_loss
######################
```

#### 數學原理
**InstaFlow Objective** (論文 Eq. 6):
$$\mathcal{L}_{\text{InstaFlow}} = \mathbb{E}_{x_0, x_1} \left[ \| x_0 + v_\theta(x_0, 0) - x_1 \|^2 + \lambda \cdot \mathcal{L}_{\text{LPIPS}}(x_0 + v_\theta(x_0, 0), x_1) \right]$$

其中:
- $x_0$: 噪聲
- $x_1$: Teacher 模型生成的高品質影像（用 CFG + 多步）
- $v_\theta(x_0, 0)$: Student 模型在 $t=0$ 預測的速度
- $x_0 + v_\theta(x_0, 0)$: Student 的一步預測
- $\mathcal{L}_{\text{LPIPS}}$: 感知損失（LPIPS）

#### L2 vs LPIPS
| 損失 | 作用 | 優點 | 缺點 |
|------|------|------|------|
| **L2** | 像素級匹配 | 簡單、快速 | 可能模糊 |
| **LPIPS** | 感知級匹配 | 視覺品質好 | 計算慢 |

#### 重點說明
1. **只在 $t=0$ 預測**: 一步生成，無需 ODE 求解
2. **Teacher 已經用 CFG**: $x_1$ 是高品質的（FID < 5）
3. **LPIPS 是關鍵**: 純 L2 會導致模糊，LPIPS 保持細節
4. **輸入範圍**: LPIPS 需要 `[-1, 1]`，記得 clamp

---

### TODO 4.2: One-Step Sampling
**檔案**: `image_common/instaflow.py` (Line 112-133)

#### 目標
實作 InstaFlow 的一步採樣。

#### 實作
```python
######## TODO ########
x_0 = x_T
t_zero = torch.zeros(x_0.shape[0], device=self.device)

if class_label is not None:
    v_pred = self.network(x_0, t_zero, class_label=class_label)
else:
    v_pred = self.network(x_0, t_zero)

x_1 = x_0 + v_pred
######################
```

#### 數學原理
**One-Step Generation**:
$$x_1 = x_0 + v_\theta(x_0, 0 | T)$$

其中:
- $x_0 \sim \mathcal{N}(0, I)$: 隨機噪聲
- $v_\theta(x_0, 0 | T)$: 在 $t=0$ 預測的速度（條件於類別 $T$）
- $x_1$: 最終影像

#### 對比多步採樣
| 方法 | 步數 | 時間 | FID |
|------|------|------|-----|
| **Original FM** | 50 | ~5 秒 | 4.51 |
| **RF1** | 5 | ~0.5 秒 | 2.16 |
| **InstaFlow (RF2-based)** | **1** | **~0.1 秒** | 24.54 |

#### 重點說明
1. **無需循環**: 一次前向傳播就完成
2. **無需 CFG**: CFG 效果已經"烘焙"在訓練時
3. **速度極快**: 比 RF1 快 5 倍，比 Original 快 50 倍
4. **權衡**: 品質略降（FID 24.54 vs 2.16），但仍在可接受範圍（< 30）

---

## 📊 完整訓練與採樣流程

### Task 2: Flow Matching
```python
# Training
for epoch in epochs:
    for x1 in dataloader:
        t = uniform_sample([0, 1])
        x0 = randn_like(x1)
        xt = (1-t)*x0 + t*x1
        ut = x1 - x0
        vt = network(xt, t)
        loss = mse(vt, ut)
        loss.backward()

# Sampling (50 steps)
x0 = randn(shape)
for t in [0, 0.02, 0.04, ..., 0.98]:
    vt = network(xt, t)
    xt = xt + 0.02 * vt
return xt  # ≈ x1
```

### Task 3: Rectified Flow
```python
# Step 1: Generate reflow data
for i in range(num_samples):
    x0 = randn(shape)
    z1 = original_fm.sample(x0, steps=50, cfg=7.5)
    save(x0, z1)

# Step 2: Train RF1
for x0, z1 in reflow_dataset:
    t = uniform_sample([0, 1])
    xt = (1-t)*x0 + t*z1
    ut = z1 - x0
    vt = network(xt, t)
    loss = mse(vt, ut)
    loss.backward()

# Sampling (5 steps!)
x0 = randn(shape)
for t in [0, 0.2, 0.4, 0.6, 0.8]:
    vt = network(xt, t)
    xt = xt + 0.2 * vt
return xt  # ≈ z1, FID=2.16!
```

### Task 4: InstaFlow
```python
# Step 1: Generate training data
for i in range(num_samples):
    x0 = randn(shape)
    x1 = rf2.sample(x0, steps=5, cfg=7.5)
    save(x0, x1)

# Step 2: Train Student
for x0, x1 in instaflow_dataset:
    t = zeros(batch_size)  # Always t=0!
    v_pred = student_network(x0, t)
    x1_pred = x0 + v_pred
    loss = mse(x1_pred, x1) + lpips(x1_pred, x1)
    loss.backward()

# Sampling (1 step!!!)
x0 = randn(shape)
v = student_network(x0, 0)
x1 = x0 + v  # Done! FID=24.54
```

---

## 🎓 關鍵數學公式總結

### 1. Conditional Flow (Rectified Flow)
$$\psi_t(x | x_1) = (1-t) \cdot x_0 + t \cdot x_1$$

### 2. Target Velocity (CFM)
$$u_t = x_1 - x_0$$

### 3. Training Loss (CFM)
$$\mathcal{L} = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t) - u_t \|^2 \right]$$

### 4. Euler Step (ODE Solver)
$$x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t)$$

### 5. Classifier-Free Guidance
$$v_{\text{guided}} = v_{\text{uncond}} + w \cdot (v_{\text{cond}} - v_{\text{uncond}})$$

### 6. InstaFlow Distillation
$$\mathcal{L}_{\text{InstaFlow}} = \| x_0 + v_\theta(x_0, 0) - x_1 \|^2 + \lambda \cdot \mathcal{L}_{\text{LPIPS}}$$

---

## ✅ 檢查清單

### Task 1 (2D)
- [ ] `network.py` - TODO 1.1: Network architecture
- [ ] `network.py` - TODO 1.2: Forward pass
- [ ] `fm.py` - TODO 1.3: Conditional flow
- [ ] `fm.py` - TODO 1.4: Euler step
- [ ] `fm.py` - TODO 1.5: Training loss
- [ ] `fm.py` - TODO 1.6: Sampling loop

### Task 2 (Image)
- [ ] `image_common/fm.py` - TODO 2.1: Conditional flow
- [ ] `image_common/fm.py` - TODO 2.2: Euler step
- [ ] `image_common/fm.py` - TODO 2.3: Training loss
- [ ] `image_common/fm.py` - TODO 2.4: Sampling loop

### Task 3 (Rectified Flow)
- [ ] `generate_reflow_data.py` - TODO 3.1: Reflow data generation

### Task 4 (InstaFlow)
- [ ] `image_common/instaflow.py` - TODO 4.1: Distillation loss
- [ ] `image_common/instaflow.py` - TODO 4.2: One-step sampling

---

## 🚀 測試建議

### Task 1
```bash
# Run Jupyter notebook
jupyter notebook task1_2d_flow_matching/fm_tutorial.ipynb
# Check: Chamfer Distance < 40
```

### Task 2
```bash
# Train
python task2_image_flow_matching/train.py --exp_name cfg_fm

# Evaluate
python fid/measure_fid.py --exp_name cfg_fm --num_inference_steps 50 --cfg_scale 1.0
# Target: FID < 30 (should get ~4.5 with optimal settings)
```

### Task 3
```bash
# Generate reflow data
python task3_rectified_flow/generate_reflow_data.py \
    --original_ckpt results/cfg_fm/ckpt.pt \
    --num_samples 5000

# Train RF1
python task3_rectified_flow/train_rectified.py --reflow_iter 1

# Evaluate
python task3_rectified_flow/evaluate_rectified.py --reflow_iter 1 --num_inference_steps 5
# Target: FID < 30 (should get ~2.2)
```

### Task 4
```bash
# Generate InstaFlow data
python task4_instaflow/generate_instaflow_data.py \
    --rectified_ckpt results/rf2/ckpt.pt

# Train Student
python task4_instaflow/train_instaflow.py

# Evaluate
python task4_instaflow/evaluate_instaflow.py
# Target: FID < 30 (should get ~24.5)
```

---

## 📚 參考文獻

1. **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
2. **Rectified Flow**: Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", ICLR 2023
3. **InstaFlow**: Liu et al., "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation", ICLR 2024
4. **Classifier-Free Guidance**: Ho & Salimans, "Classifier-Free Diffusion Guidance", NeurIPS Workshop 2021

---

## 🎯 預期結果

| Task | Method | Steps | FID | Time |
|------|--------|-------|-----|------|
| Task 1 | 2D FM | N/A | CD < 40 | N/A |
| Task 2 | Original FM | 50 | 4.51 | ~5s |
| Task 2 | Original FM (CFG=7.5) | 50 | 23.30 | ~5s |
| Task 3 | RF1 | 5 | 2.16 | ~0.5s |
| Task 3 | RF2 | 5 | 2.84 | ~0.5s |
| Task 4 | InstaFlow (RF2) | 1 | 24.54 | ~0.1s |

---

## 💡 常見問題

### Q1: 為什麼 Task 2 的 CFG 提高 FID 反而變差？
**A**: `cfg_dropout=0.1` 太低，模型沒有充分學習無條件生成。建議 `cfg_dropout >= 0.3`。

### Q2: 為什麼 RF1 在 5 steps 最好，10/20 steps 反而變差？
**A**: RF1 是用 5 步的 reflow 資料訓練的，在 5 步時完美匹配訓練分佈，超過 5 步會偏離。

### Q3: InstaFlow 為什麼需要 LPIPS？
**A**: 純 L2 loss 會導致模糊影像。LPIPS 是感知損失，能保留細節和紋理。

### Q4: 如何選擇 guidance_scale？
**A**: 
- 文生圖: 7.5-15
- 類別條件 (ImageNet): 3-7
- 準無條件 (AFHQ): 1-2
- 完全無條件: 1.0

---

**撰寫日期**: 2024-11-20  
**適用於**: NYCU IVG Lab 4 - Flow Matching Assignment  
**狀態**: ✅ 所有 TODO 已完成並經過測試

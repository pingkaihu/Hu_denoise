# AP-BSN 演算法一致性分析報告

**對象**: `denoise_apbsn.py`
**參考論文**: Lee et al., "AP-BSN: Self-Supervised Denoising for Real-World Images via Asymmetric PD and Blind-Spot Network," CVPR 2022 ([arXiv:2203.11799](https://arxiv.org/abs/2203.11799))
**官方實作**: [github.com/wooseoklee4/AP-BSN](https://github.com/wooseoklee4/AP-BSN)
**日期**: 2026-04-14

---

## 摘要

本程式在**部分概念**上確實忠於 AP-BSN 論文（PD 的數學定義、avg_shifts 推論概念），但在**五個核心技術點**上與論文存在嚴重偏差，包括：網路架構、Blind-spot 機制、損失函數、「不對稱 PD」核心創新的缺失，以及 R3 後處理步驟的缺失。

---

## 1. 背景：AP-BSN 論文的核心貢獻

論文提出三個相互依存的設計：

| 貢獻 | 說明 |
|------|------|
| **PD (Pixel-Shuffle Downsampling)** | 將空間相關性噪聲打散為像素獨立噪聲，使 BSN 可處理真實相機噪聲 |
| **Asymmetric PD (AP)** | 訓練用大步幅 `pd_a=5`，推論用小步幅 `pd_b=2`，解決單一步幅的固有取捨 |
| **R3 (Random-Replacing Refinement)** | 推論後處理：T=8 次隨機以原始噪聲像素替換去噪結果並平均，提升品質且無額外參數 |

論文的 BSN 網路架構為 **DBSNl**：以 CentralMaskedConv2d（結構性 blind-spot 約束）搭配 dilated convolution（膨脹卷積），而非 U-Net。

---

## 2. 偏差逐點分析

### 偏差 1：「不對稱 PD」完全缺失 — Critical

**AP-BSN 的 "A" (Asymmetric) 正是論文的核心創新。**

| 面向 | 論文 | 本程式 |
|------|------|--------|
| 訓練步幅 | `pd_a = 5`（大步幅確保像素獨立性）| 同一 `pd_stride`（預設=2）|
| 推論步幅 | `pd_b = 2`（小步幅保留影像細節）| 同一 `pd_stride` |

論文說明（CVPR 2022 paper, §3.2）：

> "We introduce AP (Asymmetric PD) which uses different PD stride factors for training (pd_a) and inference (pd_b), resolving the inherent trade-off caused by a specific PD stride factor."

**影響**：大步幅訓練（pd_a=5）使 PD 域中相鄰像素的相關性大幅降低，讓 BSN 的像素獨立假設成立。若訓練步幅也設為 2，對真實相機噪聲（ISP 空間相關噪聲）效果大打折扣。本程式的推論只有 avg_shifts 機制但沒有用較小步幅推論，缺少了不對稱設計的本質。

---

### 偏差 2：網路架構 — Critical

**論文使用 DBSNl（Dilated Blind-Spot Network）；本程式使用標準 U-Net。**

官方 GitHub [`src/model/DBSNl.py`](https://github.com/wooseoklee4/AP-BSN/blob/master/src/model/DBSNl.py) 揭示：

| 面向 | 論文 DBSNl | 本程式 BSNUNet |
|------|-----------|---------------|
| 架構類型 | 兩個並行分支的殘差膨脹卷積 | 四層 Encoder-Decoder U-Net |
| 膨脹率 | 兩分支分別使用 dilation=2, dilation=3 | 無（所有 Conv2d 均 dilation=1）|
| 通道數 | base_ch=128，9 個殘差塊/分支 | base_features=32，4 層 |
| 下採樣 | 無 MaxPool，保持全解析度 | 3 次 MaxPool(2) |
| Blind-spot 層 | `CentralMaskedConv2d`（架構層級）| 無 |
| 輸入通道 | 3（sRGB）→ r² 經 PD 轉換 | r²（直接） |

官方 DBSNl 中的 blind-spot 機制（`DBSNl.py`）：

```python
class CentralMaskedConv2d(nn.Conv2d):
    def forward(self, x):
        self.weight.data[:, :, kH//2, kH//2] = 0  # 結構性遮蔽中心
        return super().forward(x)
```

本程式的評論也坦承：

```
# Blind-spot is enforced via N2V-style masking in APBSNDataset —
# NOT by architectural constraints.
```

**影響**：論文的 DBSNl 具有**架構層級的 blind-spot 保證**，即使推論時不做任何掩蔽，網路結構上就無法使用中心像素進行預測。U-Net 無此保證，若不做遮蔽訓練則會退化為一般去噪器。

---

### 偏差 3：損失函數 — Major

| 面向 | 論文 | 本程式 |
|------|------|--------|
| 損失類型 | **L₁ norm**: `‖I^s_BSN − I_N‖₁` | **MSE (L₂)**: `nn.MSELoss(reduction='sum')` |
| 目標像素 | **全部像素**（BSN output vs. 含噪輸入）| **僅遮蔽像素**（N2V 風格）|
| 參考像素 | 含噪輸入 `I_N` | 原始未遮蔽 patch 值 |

論文（§3.1）：

> "L_BSN = ‖I^s_BSN − I_N‖₁"

這是 Noise2Noise 精神的體現：預測含噪影像，讓網路對噪聲期望值趨零。本程式採 N2V 風格（只對遮蔽位置計算損失），雖在語義上類似，但：

- L₁ 對離群值更穩健，更適合真實相機噪聲（具重尾分布）
- MSE 會偏好平滑解，可能導致細節過度平滑

---

### 偏差 4：R3（Random-Replacing Refinement）完全缺失 — Major

官方 GitHub [`src/model/APBSN.py`](https://github.com/wooseoklee4/AP-BSN/blob/master/src/model/APBSN.py) 揭示 R3 的超參數：

```python
R3_p = 0.16   # 隨機替換比例
R3_T = 8      # 重複次數（T次平均）
```

**R3 演算法（推論後處理）**：
1. 取 BSN 去噪結果
2. 隨機以概率 `R3_p` 用原始含噪像素替換去噪結果
3. 再跑一次 BSN
4. 重複 T=8 次並平均

論文稱 R3「significantly improves the performance of AP-BSN without any additional parameters」。本程式的 `avg_shifts` 機制（平均 r² 個 PD shift 對齊）是個**不同**的 artifact-reduction 技術，不能替代 R3。

---

### 偏差 5：avg_shifts 實作與 AP 的「AP 意涵」混淆 — Minor

本程式將 `avg_shifts=True` 描述為「AP quality mode」，但論文的「AP」並非指 shift averaging，而是**Asymmetric PD**（不對稱步幅）。

| 面向 | 論文 AP inference | 本程式 avg_shifts |
|------|------------------|-----------------|
| 機制 | 使用較小步幅 `pd_b=2` 推論（保留細節）| 平均 r² 個空間 shift offset |
| 消除 artifact | 透過小步幅自然避免 PD 網格效應 | 平均多 pass 消除網格效應 |
| 計算成本 | 單次前向傳播 | r² 次前向傳播（r=2: 4次，r=5: 25次）|

avg_shifts 的概念（對多個 PD offset 求平均）在論文中並非主要貢獻，但作為消除 PD 網格偽影的手段是合理的工程選擇。然而，將 `avg_shifts=True` 稱為「AP quality mode」在語義上具有誤導性，因為 AP 的真正含義是不對稱步幅的設計。

---

## 3. 符合論文之處

| 面向 | 狀況 |
|------|------|
| PD 的數學定義 | ✅ 正確實作 `pd_downsample`（phase-offset 分組）|
| PD 逆操作 | ✅ `pd_upsample` 正確還原空間排列 |
| numpy PD 預計算 | ✅ `_numpy_pd` 數學上等價 |
| Reflect padding | ✅ 論文與實作均使用 reflect padding 處理非整倍數 |
| 自監督訓練精神 | ✅ 不需乾淨影像，從噪聲影像本身訓練 |
| Adam 優化器 | ✅ 使用 Adam（與論文預設相符）|

---

## 4. 各偏差嚴重度摘要

| # | 偏差項目 | 嚴重度 | 影響 |
|---|---------|--------|------|
| 1 | Asymmetric PD 缺失 | Critical | 論文核心創新完全缺失；對空間相關噪聲效果下降 |
| 2 | 架構：UNet vs DBSNl | Critical | 無結構性 blind-spot 保證；感受野與論文不同 |
| 3 | 損失函數 L₁ vs MSE | Major | 對真實噪聲的重尾分布處理次優 |
| 4 | R3 缺失 | Major | 論文稱其「顯著提升」的後處理步驟缺席 |
| 5 | avg_shifts 命名混淆 | Minor | 功能合理但語義誤導 |

---

## 5. 對 SEM 應用的特殊評估

SEM 噪聲（Poisson + Gaussian 加性噪聲）是**像素獨立**的，與真實相機噪聲（空間相關 ISP 噪聲）性質不同。這個差異影響嚴重度評估：

- **Asymmetric PD 缺失**：對 SEM 而言影響相對較小，因為 SEM 噪聲本身已是像素獨立的，小步幅 pd_stride=2 就足以讓 N2V 假設成立。若改為針對真實相機噪聲使用本程式，則影響嚴重。
- **U-Net vs DBSNl**：對 SEM 的 N2V 風格遮蔽訓練，U-Net 功能上仍可用。論文原設計為 RGB 相機影像；灰階 SEM 用 U-Net 是合理的適應性簡化。
- **MSE vs L₁**：SEM 噪聲相對高斯，MSE 在此場景下損失不如 L₁ 大。

---

## 6. 結論

`denoise_apbsn.py` 在**標題與名稱**上宣稱實作 AP-BSN，但實質上是一個：

- 以 **N2V 風格遮蔽訓練** 取代論文的架構式 blind-spot
- 以 **U-Net** 取代論文的 DBSNl 膨脹卷積
- 使用 **MSE** 而非論文的 L₁ 損失
- **完全缺少** Asymmetric PD（論文命名之由來）與 R3 後處理

的客製化去噪器。它更接近於「PD 預處理 + N2V 風格訓練 + U-Net」的組合，而非真正的 AP-BSN。

對 SEM 這種像素獨立噪聲場景，此簡化實作仍可產生合理的去噪效果，但若要比較文獻中 AP-BSN 的性能數據，需注意這並非同一演算法。

---

## 參考資料

- Lee et al., [AP-BSN: Self-Supervised Denoising for Real-World Images via Asymmetric PD and Blind-Spot Network](https://arxiv.org/abs/2203.11799), CVPR 2022
- [CVPR 2022 Official Paper (PDF)](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_AP-BSN_Self-Supervised_Denoising_for_Real-World_Images_via_Asymmetric_PD_and_CVPR_2022_paper.pdf)
- [Official GitHub: wooseoklee4/AP-BSN](https://github.com/wooseoklee4/AP-BSN)
- [DBSNl Source Code](https://github.com/wooseoklee4/AP-BSN/blob/master/src/model/DBSNl.py)
- [APBSN Model Wrapper](https://github.com/wooseoklee4/AP-BSN/blob/master/src/model/APBSN.py)

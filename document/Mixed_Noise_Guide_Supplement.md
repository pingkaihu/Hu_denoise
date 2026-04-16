# Mixed_Noise_Speckle_ShotNoise_Guide 補充分析
### 批判性審查、遺漏方法、優先實作清單

> **閱讀前提：** 本文件是 `Mixed_Noise_Speckle_ShotNoise_Guide.md` 的補充，
> 不重複原文內容，僅記錄原文的論據缺陷與未涵蓋方法。

---

## 目錄

1. [原文內部邏輯裂縫](#1-原文內部邏輯裂縫)
2. [原文未提及的可行方法](#2-原文未提及的可行方法)
3. [決策樹空白地帶](#3-決策樹空白地帶)
4. [優先實作清單](#4-優先實作清單)
5. [補充參考資源](#5-補充參考資源)

---

## 1. 原文內部邏輯裂縫

### 1.1 PN2V 首選論據不完整：GMM 容量問題

原文聲稱 GMM 可「直接對 Poisson-Gamma 混合建模」，但未討論容量限制：

```
Poisson(λ) 的離散 PMF 與 Gamma(L, 1/L) 的連續 PDF 卷積
沒有封閉解，混合分佈的尾部行為複雜。

原文建議 num_gaussians=3，然而：
  - 當 L 小（強 speckle，L < 3）時，混合分佈呈重尾
  - 3 個高斯的覆蓋能力有限，尾部擬合不足
  - num_gaussians 超參數的敏感性完全未討論
```

**行動建議：** 對強 speckle 場景（ENL < 3），在訓練前先驗證
`num_gaussians` 從 3 增加到 5–7 是否改善 GMM 擬合品質。

---

### 1.2 GR2R 第二優先的理論支撐有誤

原文將 GR2R 定位為「Gamma 噪聲最直接」方案，但 Binomial splitting 對混合噪聲的有效性有根本問題：

```
GR2R (Monroy et al., CVPR 2025) 的 Binomial splitting：
  給定一個 Poisson(λ) 樣本 Y，
  用 Binomial(Y, p) 可分裂為兩個獨立 Poisson pair → 有效 N2N pair

混合噪聲中：y = Poisson(x · n_speckle)
  n_speckle 對同一次曝光是同一個實現
  → 分裂後的兩份雖 Poisson 部分獨立
  → 但兩份共享同一個 n_speckle（相關！）
  → 不再構成有效的 N2N pair

結論：GR2R 對純 shot noise (Poisson) 有效，
     對 Poisson-Gamma 混合的「直接支援」
     在理論上沒有嚴格依據。
```

**影響：** 原文的優先順序「GR2R > Self2Self」在混合場景下需要重新評估。
Self2Self（無假設）反而比 GR2R 對混合噪聲更保守穩健。

---

### 1.3 grain > 2px 門檻無理論根據

原文多處用 2px 作為分支點，但：

- N2V blind-spot 排除中心 1px，grain 超過 **1px** 時空間相關性假設已開始失效
- 2px 門檻沒有引用任何理論分析或消融實驗
- 實際上門檻取決於 blind-spot 半徑與 grain 的自相關長度之比

**修正建議：** 將門檻改為「grain size 是否明顯超過 blind-spot 半徑（通常 1px）」，即 grain ≥ 2px 就需要注意，grain ≥ 4px 才達到 Self2Self 的適用範圍。

---

### 1.4 ENL 估計的可靠性被過度樂觀化

原文的 ENL 公式：
```python
L = (mean_val ** 2) / (var_val + 1e-8)
```

假設所選 ROI 是均勻（homogeneous）區域，但：

| 問題 | 後果 |
|---|---|
| ROI 包含材料邊緣或微結構 | ENL 嚴重低估（var 被信號變異拉高）|
| SEM 帶電效應造成亮度梯度 | mean 估計不代表真實信號 |
| 背景均勻區面積小於 50×50px | 統計量方差大，估計不穩定 |

**行動建議：** 用多個候選 ROI 取中位數 ENL，並設定估計信賴區間。
ENL 方差 > 30% 時不應使用 GR2R。

---

### 1.5 DIP 替代損失函數公式有誤

原文的 `mixed_noise_loss` 中 Gamma NLL 項：

```python
# 原文（有誤）
gamma_nll = (L - 1) * torch.log(output) + L * target / output
```

**問題：** speckle 模型是 `y = x · n`，`n ~ Gamma(L, L)`。
`output` 代表信號估計 `x̂`，`target` 代表觀測 `y`。
應針對 `n̂ = target / output` 套用 Gamma 分佈導出 NLL：

```python
# 正確版本（對 n = y/x 的 Gamma(L, L) NLL）
ratio = target / (output + eps)
gamma_nll = (1 - L) * torch.log(ratio) + L * ratio
# 等價於：
# gamma_nll = (L - 1) * torch.log(output / target) + L * target / output
```

原文版本在 `L > 1` 時梯度方向對高強度像素有系統性偏移。

---

## 2. 原文未提及的可行方法

### 2.1 Noise2Score — 最具補充價值

**論文：** Kim et al., NeurIPS 2022  
**核心：** Tweedie's formula 的廣義化

```
對高斯噪聲（Noise2Score 的基礎）：
  E[x|y] = y + σ² ∇_y log p(y)
        信號估計  觀測  score function

對指數族噪聲，等效公式成立，需要估計 ∇_y log p(y)
Poisson 和 Gamma 均屬指數族 → 理論完備

核心優勢（相較於 PN2V）：
  ✅ 不需要 blind-spot，沒有像素獨立假設
  ✅ 不需要 GAT，原始空間直接工作
  ✅ 不需要指定 GMM 容量（num_gaussians）
  ✅ 對重尾分佈的泛化能力優於有限混合模型
  ✅ 混合噪聲可透過對應混合指數族的 score function 處理
```

**實作難度：** 中等。需要訓練一個 score estimator（類似 denoising score matching），
但不依賴任何特定架構，可直接套用現有 UNet。

---

### 2.2 AP-BSN — 項目已實作但未納入文件

本項目已有 [denoise_apbsn_lee.py](../denoise_apbsn_lee.py) 和
[denoise_apbsn_faithful.py](../denoise_apbsn_faithful.py)，
但 `Mixed_Noise_Speckle_ShotNoise_Guide.md` 的決策樹完全未提及。

```
AP-BSN 對混合噪聲的關鍵能力：

Pixel-Shuffle Downsampling (PD)：
  stride=2 → 把 grain size 2px 的空間相關噪聲
            轉為像素獨立（每個 PD 輸出 channel 只含獨立像素）
  stride=3 → 對應 grain size ≤ 3px

適用範圍（比 N2V 寬，不需要 Self2Self 的長訓練時間）：
  ✅ grain 2–4px（N2V 失效但 Self2Self 過慢的區間）
  ✅ 混合噪聲（無噪聲分佈假設）
  ✅ R3 refinement 可有效抑制棋盤格效應
  ✅ 訓練可重用（一次訓練批量推論）

建議補入決策樹的位置：
  「grain 2-4px」分支，優先於 Self2Self（訓練時間長）
```

---

### 2.3 Blind2Unblind — N2V 到 Self2Self 之間的橋梁

**論文：** Wang et al., CVPR 2022

```
N2V 的核心限制：
  blind-spot 半徑 = 1px
  → 鄰近像素的 speckle 相關性仍洩漏給預測器
  → grain 2px 時已有明顯偏差

Blind2Unblind 的改進：
  引入「re-visible」loss：部分時間恢復 blind-spot，
  讓模型利用全域資訊輔助局部噪聲判斷
  → 訓練目標放寬了像素間完全獨立的假設
  → 對 grain 2–4px 效果顯著優於 N2V

決策樹的填補位置：
  grain 1-2px：N2V / PN2V（原文覆蓋）
  grain 2-4px：Blind2Unblind（原文空白）← 補入
  grain 4px+：AP-BSN / Self2Self（原文覆蓋）
```

---

### 2.4 Haar-Fisz 轉換 — 低計數場景的 GAT 替代

**論文：** Fryzlewicz & Nason, 2004

```
GAT 在極低計數（background < 0.02）時失效的根本原因：
  sqrt(x + 3/8) ≈ sqrt(x) 在 x → 0 時一階近似不準

Haar-Fisz 的優勢：
  ✅ 不需要估計 gain 或 sigma_read
  ✅ 利用 Haar 小波係數的方差穩定性（無需參數）
  ✅ 在極低計數場景穩健性優於 GAT
  ✅ 可與 BM3D 或 N2V 組合

文件空缺：
  原文在「低計數場景」直接跳到 PN2V 原始空間（需訓練），
  沒有快速傳統基準替代 GAT+BM3D（已失效）。
  Haar-Fisz + BM3D 填補這個空缺。
```

---

### 2.5 Plug-and-Play ADMM（PnP）— 精確 NLL + 任意先驗

```
框架：
  argmin_x  [-log p(y|x)]  +  λ·R(x)
              Poisson/Gamma     深度去噪先驗
              精確 NLL          (任意 DnCNN/UNet)

ADMM 解耦：
  子問題 1：數據保真（有解析解，精確反映物理模型）
  子問題 2：去噪（任意強去噪器插入，即 "plug-and-play"）

對比 DIP：
  ✅ 不需要早停（優化目標有明確收斂保證）
  ✅ 先驗可替換，不鎖定在單一架構
  ✅ 數據保真項精確，不做 GAT 近似

代表論文：
  Rond et al., Poisson Image Denoising using BM3D, 2016
  Teodoro et al., Image restoration and reconstruction using PnP, 2019
```

---

### 2.6 CVF-SID — 無假設成分分解

**論文：** Neshatavar et al., CVPR 2022

```
思路：把含噪影像分解為「訊號分支」和「噪聲分支」
     利用噪聲的統計性質（均值零、空間獨立）作為自監督信號
     不需要 blind-spot，不需要噪聲分佈假設

對混合噪聲的優勢：
  ✅ 乘性噪聲（speckle）的分解比 blind-spot 方法更自然
  ✅ 比 Self2Self 訓練時間短（不需 450k 步）
  ✅ 無需任何參數估計（ENL、gain、sigma_read）

定位：grain 大（> 4px）且不想訓練 90 分鐘的 Self2Self 時的替代
```

---

### 2.7 HDN（Hierarchical DivNoising）— PN2V 的多尺度延伸

**論文：** Prakash et al., NeurIPS 2021

```
PN2V：單一 blind-spot UNet + GMM
HDN：層級 VAE（多尺度 latent）+ GMM 噪聲模型

多尺度先驗的優勢：
  ✅ 可同時在粗（grain 結構）和細（像素）尺度建模
  ✅ 對 speckle grain 明顯的場景，信號先驗更豐富
  ✅ 在 PN2V GMM 擬合不足時（L 小、num_gaussians 不夠）可提升效果

定位：PN2V GMM 擬合效果不佳時的升級方案
```

---

## 3. 決策樹空白地帶

原文決策樹在以下象限未覆蓋推薦方法：

```
             grain size
                 ↑
        大(>4px) │ Self2Self    CVF-SID ← 原文空白
                 │ GR2R*
        中(2-4px)│ Blind2Unblind←      AP-BSN ← 原文空白（但已實作！）
                 │ 原文空白 ↑
        小(<2px) │ N2V / PN2V   N2V / PN2V
                 └──────────────────────────────→
                  Shot >> Speckle     混合均衡

* GR2R 在混合場景的理論支撐見 §1.2 的批評
```

**修訂後的決策樹補充分支（接在原文 Step 2 之後）：**

```
└─ 混合型 → grain size 評估
      ├─ grain < 2px
      │    └─► PN2V 原始空間（首選）/ GAT+N2V（快速）
      │
      ├─ grain 2–4px ← 原文空白，補入：
      │    ├─► AP-BSN（已實作，pd_stride=2 覆蓋 2px grain）
      │    └─► Blind2Unblind（可替換 blind-spot 訓練流程）
      │
      └─ grain > 4px
           ├─ 能接受 30–90 分鐘/張 → Self2Self
           ├─ 希望更快 → CVF-SID
           └─ 能估計 ENL + 純 shot noise → GR2R
                （混合場景慎用，見 §1.2）
```

---

## 4. 優先實作清單

依「理論收益 / 實作成本」排序：

---

### 優先 A — 無需新模型，修補現有代碼

#### A1. 修正 DIP 損失函數中的 Gamma NLL 公式

**影響：** `denoise_DIP.py` 的混合損失模式  
**工作量：** 3 行改動  
**位置：** 參照 §1.5 的正確公式

```python
# 替換 denoise_DIP.py 中的 gamma_nll 行：
ratio = target / (output + eps)
gamma_nll = (1 - L) * torch.log(ratio + eps) + L * ratio
```

---

#### A2. 補充 AP-BSN 進混合噪聲決策樹

**影響：** `Mixed_Noise_Speckle_ShotNoise_Guide.md` 的完整性  
**工作量：** 文件修訂，不涉及代碼  
**內容：** 在 grain 2–4px 分支加入「AP-BSN（`pd_stride=2` 或 `3`）」，
指向 `denoise_apbsn_lee.py` / `denoise_apbsn_lee_multi.py`

---

#### A3. PN2V 的 num_gaussians 診斷工具

**影響：** 強 speckle（ENL < 3）場景的 GMM 擬合品質  
**工作量：** 在 `denoise_PN2V.py` 加入 GMM log-likelihood 評估，
比較 `num_gaussians=3, 5, 7` 的 BIC（貝葉斯信息準則）

```python
def select_num_gaussians(pixel_pairs, candidates=[3, 5, 7]):
    """用 BIC 自動選擇最佳 GMM 容量"""
    from sklearn.mixture import GaussianMixture
    best_bic, best_k = np.inf, 3
    for k in candidates:
        gmm = GaussianMixture(n_components=k).fit(pixel_pairs)
        bic = gmm.bic(pixel_pairs)
        if bic < best_bic:
            best_bic, best_k = bic, k
    return best_k
```

---

### 優先 B — 新腳本，高理論收益

#### B1. Noise2Score 實作腳本

**對應場景：** PN2V GMM 容量不足、ENL < 3 的強 speckle + shot 混合  
**理論依據：** 指數族 Tweedie 公式，對 Poisson-Gamma 嚴謹  
**工作量：** 中等（訓練 score estimator，可重用現有 UNet backbone）

建議腳本名：`denoise_N2Score.py`

關鍵實作點：
- 訓練目標：denoising score matching（DSM loss）
- 推論：`x̂ = y + σ²(y) · score_net(y)`，其中 `σ²(y)` 由噪聲模型決定
- 對 Poisson：`σ²(y) = y`（方差等於均值）
- 對 Gamma(L)：`σ²(y) = y²/L`

---

#### B2. Haar-Fisz + BM3D 快速基準腳本

**對應場景：** 低計數（background < 0.02），需要 < 1 分鐘的快速基準  
**填補空缺：** GAT+BM3D 在低計數失效後的替代  
**工作量：** 低（PyWavelets + bm3d，< 50 行）

建議腳本名：`denoise_HaarFisz_BM3D.py`

```python
import pywt, bm3d, numpy as np

def haar_fisz_transform(image):
    """Haar-Fisz 方差穩定化（無需任何參數估計）"""
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs
    # Fisz 正規化：detail / sqrt(approx)
    eps = 1e-8
    cH_f = cH / (np.sqrt(np.abs(cA)) + eps)
    cV_f = cV / (np.sqrt(np.abs(cA)) + eps)
    cD_f = cD / (np.sqrt(np.abs(cA)) + eps)
    return pywt.idwt2((cA, (cH_f, cV_f, cD_f)), 'haar')

def haar_fisz_bm3d(image):
    img_hf = haar_fisz_transform(image)
    sigma_est = np.std(img_hf[:30, :30])  # 背景估計
    return bm3d.bm3d(img_hf, sigma_psd=sigma_est)
```

---

#### B3. AP-BSN 混合噪聲模式文件整合

**對應場景：** grain 2–4px 混合噪聲，需要批量可重用模型  
**工作量：** 低（現有腳本已完整，只需加入決策樹和 CLAUDE.md）

在 CLAUDE.md 的「Noise Type Decision」加入：
```
- Mixed noise, grain 2-4px → denoise_apbsn_lee.py (pd_stride=2)
```

---

### 優先 C — 長期研究項目

| 項目 | 說明 | 工作量 |
|---|---|---|
| **Blind2Unblind** | 修改 N2V 訓練使其容忍 2-4px grain | 高（需改訓練架構）|
| **PnP-ADMM** | 精確 Poisson/Gamma NLL + UNet 先驗 | 高（優化框架）|
| **CVF-SID** | 成分分解，替代大 grain 場景的 Self2Self | 高（新架構）|
| **HDN** | PN2V 多尺度延伸，強 speckle 場景 | 高（VAE 訓練）|

---

## 5. 補充參考資源

| 論文 | 方法 | 對本項目的價值 |
|---|---|---|
| Kim et al., NeurIPS 2022 | Noise2Score (Tweedie) | 對 Poisson-Gamma 比 GMM 更嚴謹 |
| Wang et al., CVPR 2022 | Blind2Unblind | 填補 grain 2–4px 空缺 |
| Neshatavar et al., CVPR 2022 | CVF-SID | 無假設成分分解 |
| Prakash et al., NeurIPS 2021 | HDN | PN2V 多尺度延伸 |
| Fryzlewicz & Nason, 2004 | Haar-Fisz | 低計數 VST 替代 GAT |
| Rond et al., 2016 | PnP-ADMM (Poisson) | 精確 NLL + 任意先驗 |
| Teodoro et al., 2019 | PnP 混合噪聲 | PnP 框架對混合噪聲的延伸 |

---

*本補充文件基於對 `Mixed_Noise_Speckle_ShotNoise_Guide.md` 的批判性審查，
撰寫日期：2026-04-16。*

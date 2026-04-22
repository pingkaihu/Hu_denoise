# AP-BSN 演算法實作比較報告

**對象**: `denoise_apbsn.py` · `denoise_apbsn_faithful.py` · `denoise_apbsn_lee.py`
**參考論文**: Lee et al., "AP-BSN: Self-Supervised Denoising for Real-World Images via Asymmetric PD and Blind-Spot Network," CVPR 2022 ([arXiv:2203.11799](https://arxiv.org/abs/2203.11799))
**官方實作**: [github.com/wooseoklee4/AP-BSN](https://github.com/wooseoklee4/AP-BSN)
**最後更新**: 2026-04-15

---

## 摘要

本報告比較三個本地 AP-BSN 實作與論文、官方程式碼之間的一致性。

| 腳本 | 定位 | 論文符合度 |
|------|------|-----------|
| `denoise_apbsn.py` | 初版，PD + N2V-style UNet | 概念部分正確，五個核心偏差 |
| `denoise_apbsn_faithful.py` | 論文忠實版，補足所有偏差 | 高度符合論文設計 |
| `denoise_apbsn_lee.py` | 官方 code 移植版 | 最接近官方 GitHub 實作 |

---

## 1. 論文核心貢獻

論文提出三個相互依存的設計：

| 貢獻 | 說明 |
|------|------|
| **PD (Pixel-Shuffle Downsampling)** | 將空間相關性噪聲打散為像素獨立噪聲，使 BSN 可處理真實相機噪聲 |
| **Asymmetric PD (AP)** | 訓練用大步幅 `pd_a=5`，推論用小步幅 `pd_b=2`，解決單一步幅的固有取捨 |
| **R3 (Random-Replacing Refinement)** | 推論後處理：T=8 次隨機以原始噪聲像素替換去噪結果並平均，提升品質且無額外參數 |

論文的 BSN 網路架構為 **DBSNl**：以 `CentralMaskedConv2d`（結構性 blind-spot）搭配 dilated convolution，而非 U-Net。

---

## 2. `denoise_apbsn.py` — 偏差分析

### 偏差 1：Asymmetric PD 完全缺失 — Critical

AP-BSN 的 "A" 正是論文的核心創新：訓練與推論使用不同 PD 步幅。

| 面向 | 論文 | 本程式 |
|------|------|--------|
| 訓練步幅 | `pd_a = 5` | 同一 `pd_stride`（預設=2）|
| 推論步幅 | `pd_b = 2` | 同一 `pd_stride` |

對 SEM（像素獨立噪聲）影響相對較小；對真實相機噪聲（空間相關 ISP 噪聲）效果大打折扣。

### 偏差 2：網路架構 — Critical

| 面向 | 論文 DBSNl | 本程式 BSNUNet |
|------|-----------|---------------|
| 架構類型 | 兩並行分支殘差膨脹卷積 | 四層 Encoder-Decoder U-Net |
| Blind-spot 層 | `CentralMaskedConv2d`（架構層級）| 無（訓練時 N2V masking 代替）|
| 下採樣 | 無 MaxPool，保持全解析度 | 3 次 MaxPool(2) |
| 膨脹率 | dilation=2 與 dilation=3 | 無（dilation=1）|

### 偏差 3：損失函數 — Major

| 面向 | 論文 | 本程式 |
|------|------|--------|
| 損失類型 | L₁：`‖I^s_BSN − I_N‖₁` | MSE (L₂)：`nn.MSELoss` |
| 目標像素 | **全部像素** | **僅遮蔽像素**（N2V 風格）|

### 偏差 4：R3 完全缺失 — Major

本程式以 `avg_shifts`（平均 r² 個 PD offset）代替 R3，兩者是不同機制：

| 面向 | 論文 R3 | 本程式 avg_shifts |
|------|---------|-----------------|
| 作用時機 | 推論後處理，多次注入噪聲再去噪 | 推論時平均多個 PD shift 對齊結果 |
| 消除對象 | 噪聲殘留 | PD 網格偽影 |
| 計算成本 | T=8 次 BSN pass | r² 次前向傳播 |

### 偏差 5：avg_shifts 命名混淆 — Minor

`avg_shifts=True` 被描述為「AP quality mode」，但論文的「AP」指 Asymmetric PD（不對稱步幅），非 shift averaging。功能合理，語義誤導。

### 符合論文之處

| 面向 | 狀況 |
|------|------|
| PD 的數學定義 | ✅ phase-offset 分組正確 |
| PD 逆操作 | ✅ `pd_upsample` 數學等價 |
| Reflect padding | ✅ 非整倍數處理正確 |
| 自監督精神 | ✅ 不需乾淨影像 |
| Adam 優化器 | ✅ 與論文預設相符 |

### 各偏差嚴重度摘要

| # | 偏差項目 | 嚴重度 |
|---|---------|--------|
| 1 | Asymmetric PD 缺失 | Critical |
| 2 | 架構：UNet vs DBSNl | Critical |
| 3 | 損失函數 L₁ vs MSE | Major |
| 4 | R3 缺失 | Major |
| 5 | avg_shifts 命名混淆 | Minor |

---

## 3. `denoise_apbsn_faithful.py` — 論文忠實版

### 3.1 修正項目對照

| 偏差 | 修正方式 |
|------|---------|
| Asymmetric PD 缺失 | `--pd_a`（訓練）與 `--pd_b`（推論）分開設定 |
| 架構 | 實作 DBSNl（CentralMaskedConv2d + DCl + DC_branchl）|
| 損失函數 | L₁ loss，對全部像素計算 |
| R3 缺失 | 實作 R3（T=8, p=0.16），Python/numpy 實現 |

### 3.2 DBSNl 實作方式

DBSNl 的 blind-spot 由 `CentralMaskedConv2d` 在架構層級保證：

```python
class CentralMaskedConv2d(nn.Conv2d):
    def forward(self, x):
        # 建立新 tensor（不改動 weight 本體）
        return F.conv2d(x, self.weight * self._cmask, ...)
```

mask 形狀為 `(1, 1, kH, kW)`（可廣播），中心位置 `[kH//2, kW//2] = 0`。

### 3.3 PD 實作方式（numpy，CPU 預計算）

`_numpy_pd(image, r)` 將影像分解為 r² 個獨立子影像（phase offset 分組）：

```
(H, W) → (r², H//r, W//r)
```

訓練時從 r² 個子影像取 patch；推論時批次處理 r² 個子影像後再拼回。

### 3.4 R3 實作方式

```
D₀ = infer(image, pd=pd_b)
for t = 1 … T-1:
    M ~ Bernoulli(p=0.16)
    D_t = infer((1−M)·D₀ + M·image, pd=pd_b)
output = mean(D₀ … D_{T-1})
```

每次 R3 pass 都走完整 PD → BSN → PD⁻¹ 流程。

### 3.5 與 `denoise_apbsn_lee.py` 的差異預告

`faithful.py` 是論文層面的忠實移植；`lee.py` 是官方 GitHub code 層面的忠實移植。兩者在以下細節有所不同（詳見第 4 節）：PD 的運算位置、R3 是否包含 PD、Dataset 設計。

---

## 4. `denoise_apbsn_lee.py` — 官方 Code 移植版

### 4.1 與官方 GitHub 的對應關係

| 本腳本組件 | 對應官方檔案 |
|-----------|------------|
| `pixel_shuffle_down_sampling` | `src/util/util.py` |
| `pixel_shuffle_up_sampling` | `src/util/util.py` |
| `CentralMaskedConv2d`, `DCl`, `DC_branchl`, `DBSNl` | `src/model/DBSNl.py` |
| `APBSN` wrapper（含 `forward` 與 `denoise`） | `src/model/APBSN.py` |
| `self_L1` loss | `src/loss/recon_self.py` |

### 4.2 核心架構差異：PD 在模型內部

`faithful.py` 將 PD 作為資料預處理（numpy，CPU）：

```
Dataset → PD sub-images → patches → DBSNl → loss
```

`lee.py` 將 PD 包在 `APBSN.forward()` 內（官方設計）：

```
Dataset → raw crops → APBSN.forward() → [PD + DBSNl + PD⁻¹] → loss
```

官方這樣設計的好處：模型自身封裝完整，`forward()` 的輸入與輸出都在原始影像空間，不需要外部處理 PD。

### 4.3 `pixel_shuffle_down_sampling` — 官方 util.py

官方的 PD 函數與 `faithful.py` 的 numpy PD **數學等價**，但實作路徑不同：

```python
def pixel_shuffle_down_sampling(x, f, pad=0, pad_value=0.):
    b, c, H, W = x.shape
    unshuffled = F.pixel_unshuffle(x, f)          # (B, C·f², H//f, W//f)
    if pad != 0:
        unshuffled = F.pad(unshuffled, (pad,pad,pad,pad), value=pad_value)
    Hp, Wp = H//f + 2*pad, W//f + 2*pad
    return (unshuffled
            .view(b, c, f, f, Hp, Wp)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(b, c, f*Hp, f*Wp))
```

輸出空間大小 = `(H + 2·f·pad, W + 2·f·pad)`（`pad=0` 時與輸入同大小）。與 `faithful.py` 的「分解為 r² 個子影像」不同：此版本將所有 f² 個 phase offset 的像素重新排列至單一影像，保留空間鄰近性結構。

### 4.4 官方 R3 邏輯——R3 pass 不走 PD

`faithful.py` 的 R3 pass 走完整 PD → BSN → PD⁻¹：

```python
# faithful.py: R3 via _infer_single_pass (includes PD)
Dt = _infer_single_pass(model, mixed, pd=pd_b, device)
```

官方 `APBSN.denoise()` 的 R3 pass **直接呼叫 BSN**，跳過 PD：

```python
# lee.py (official): R3 via self.bsn() directly, no PD
tmp = img_pd_bsn.clone()
tmp[mask] = x[mask]               # re-inject noisy pixels
denoised[..., t] = self.bsn(tmp)  # BSN directly, NO pixel_shuffle
```

原理：第一次 PD pass（`img_pd_bsn`）的輸出已是大致乾淨的影像，R3 輸入的噪聲相關性已被打散；後續 pass 的 BSN 不需要再做 PD。

### 4.5 DC_branchl 層序 — 官方正確版本

| 位置 | 官方 DBSNl.py | 本腳本 |
|------|--------------|--------|
| CentralMaskedConv 後 | **ReLU**（無 1×1）| **ReLU**（無 1×1）✅ |
| 後續 1×1 conv 數量 | 2 組（各帶 ReLU）| 2 組（各帶 ReLU）✅ |

完整序列（stride ∈ {2, 3}）：
```
CentralMaskedConv(k=2·stride−1) → ReLU
→ [Conv1×1 + ReLU] × 2
→ DCl(dilation=stride) × num_module
→ Conv1×1 + ReLU
```

### 4.6 `CentralMaskedConv2d` 實作差異

| 面向 | 官方 | lee.py |
|------|------|--------|
| mask 形狀 | `(out_ch, in_ch, kH, kW)`（與 weight 同形）| `(1, 1, kH, kW)`（可廣播）|
| 應用方式 | `self.weight.data *= self.mask`（in-place）| `self.weight * self._cmask`（新 tensor）|
| 欄位 index | `kH//2`（官方程式有 typo，對正方形 kernel 無影響）| `kW//2`（正確）|

兩者結果完全等價，lee.py 的廣播 mask + 非 in-place 對梯度計算更乾淨。

### 4.7 與官方的剩餘差異

| 項目 | 官方 | lee.py | 影響 |
|------|------|--------|------|
| 訓練資料規模 | 大型資料集（SIDD/DND）| 單張 SEM 影像 | 刻意的 SEM 適應 |
| 輸入通道 | in_ch=3（RGB）| in_ch=1（灰階）| 刻意的 SEM 適應 |
| Data augmentation | flip + 90° rotate | 無 | 訓練品質，單張影像下影響有限 |
| LR schedule | StepLR (step=8ep, γ=0.1, 20ep) | StepLR (step=ep//3, γ=0.1) | 微小 |
| `denoise()` no_grad | 由呼叫端包裝 | `@torch.no_grad()` 裝飾器 | 微小 |

---

## 5. 三腳本橫向比較

### 5.1 架構層面

| 面向 | `apbsn.py` | `faithful.py` | `lee.py` |
|------|-----------|--------------|---------|
| 網路 | BSNUNet (U-Net) | DBSNl | DBSNl |
| Blind-spot | N2V-style masking | CentralMaskedConv2d | CentralMaskedConv2d |
| 損失函數 | MSE（masked pixel）| L₁（全 pixel）| L₁（全 pixel）|
| Asymmetric PD | ✗ | ✅ pd_a ≠ pd_b | ✅ pd_a ≠ pd_b |
| R3 | ✗ | ✅（numpy, 含 PD）| ✅（torch, 不含 PD）|
| PD 位置 | 資料預處理 | 資料預處理（numpy）| 模型內部（torch）|

### 5.2 工程層面

| 面向 | `apbsn.py` | `faithful.py` | `lee.py` |
|------|-----------|--------------|---------|
| Dataset | PD sub-image pool | PD sub-image pool | raw crop pool |
| Inference 方式 | avg_shifts 或單次 | `_infer_single_pass` + numpy R3 | `APBSN.denoise()` torch R3 |
| Tiled inference | 無（OOM 風險）| 無 | 無 |
| pd_pad 支援 | ✗ | ✗（pd_pad=0 only）| ✅ |

---

## 6. 計算資源分析

### 6.1 模型參數量

| 模型 | 參數量 |
|------|--------|
| N2VUNet (base_features=32) | ~1,790K |
| DBSNl (base_ch=64，SEM 預設) | ~915K |
| DBSNl (base_ch=128，論文預設) | ~3,660K |

DBSNl (base_ch=64) 的**參數量**約為 N2V UNet 的一半，但這並不代表計算量較少（見 6.2）。

### 6.2 訓練 FLOPs 分析

DBSNl 與 N2V UNet 的關鍵架構差異影響計算量：

**N2V UNet — 空間 pyramid 壓縮計算**

```
64×64 → enc1(32ch) → pool
32×32 → enc2(64ch) → pool
16×16 → enc3(128ch) → pool
 8×8  → enc4(256ch)         ← 最重的 conv 在此，但只有 8×8 解析度
```

enc4 的 conv3×3（256ch @ 8×8）：`FLOPs ≈ 2 × 256² × 9 × 64 ≈ 75M MACs`

**DBSNl — 全程保持原始解析度**

DBSNl **不能做 downsampling**（pool 後 blind-spot property 失效），因此：

```
64×64 → head → branch1(64ch) → branch2(64ch) → tail
        全程 64×64，無 pool
```

每個 DCl 的 conv3×3（64ch @ 64×64）：`FLOPs ≈ 2 × 64² × 9 × 64² ≈ 302M MACs`

兩個 branch 各有 9 個 DCl = **18 個 302M MACs 的 conv**：`18 × 302M = 5,436M MACs`

**每張 64×64 patch 的 MACs 估算**

| 模型 | MACs / patch |
|------|-------------|
| N2V UNet | ~1,000M |
| DBSNl (base_ch=64) | ~7,500M |
| 比值 | **7.5×** |

**每個 epoch 的總計算量**（patches/epoch = 1,800）

| 模型 | batch_size | batches/epoch | MACs/epoch |
|------|-----------|--------------|-----------|
| N2V | 128 | 14 | **1.8T** |
| AP-BSN lee | 32 | 56 | **13.5T** |
| 比值 | | | **7.5×** |

DBSNl per-epoch 計算量比 N2V 重約 7.5 倍，實測訓練時間 2× 以上，與此分析相符（GPU 對 1×1 conv 有良好 throughput 優化，使理論差距在實際上有所縮小）。

### 6.3 推論資源

| 項目 | N2V | PN2V | AP-BSN (no_r3) | AP-BSN (R3, T=8) |
|------|-----|------|---------------|-----------------|
| 推論次數 | 1 | 1 | 1 | **9** |
| 是否 tiled | ✅ | ✅ | ✗ | ✗ |
| 512×512 相對時間 | 基準 | ~1.1× | ~0.7× | **~5–6×** |
| 大影像（>1024px）| 安全 | 安全 | ⚠️ OOM 風險 | ⚠️ OOM 風險 |

AP-BSN 的 R3（T=8）使推論成本為 N2V 的 5–6 倍，且全圖推論對大影像有 OOM 風險。若影像 >1024px，建議使用 `--no_r3` 或手動分塊。

### 6.4 選用建議

| 情境 | 推薦腳本 |
|------|---------|
| GPU 記憶體 <4GB 或影像 >1024px | N2V / PN2V（tiled，安全）|
| 速度優先，均勻噪聲 | `denoise_N2V_test.py` |
| 品質優先，Poisson/Gaussian 混合噪聲 | `denoise_N2V_GMM.py` |
| 相機 sRGB 空間相關噪聲 | `denoise_apbsn_lee.py --pd_a 5 --pd_b 2 --pd_pad 2` |
| AP-BSN 快速 preview | `denoise_apbsn_lee.py --no_r3 --base_ch 32 --num_module 5` |

---

## 7. SEM 應用特殊評估

SEM 噪聲（Poisson + Gaussian 加性噪聲）為**像素獨立**，與真實相機噪聲（ISP 空間相關）性質不同，影響各偏差的嚴重度：

| 偏差 | 對相機噪聲 | 對 SEM 噪聲 |
|------|----------|-----------|
| Asymmetric PD 缺失 | 嚴重（pd_a=5 才能打散相關性）| 影響小（SEM 噪聲本身已像素獨立，pd=2 足夠）|
| UNet vs DBSNl | 嚴重（感受野與 blind-spot 保證不同）| 中等（N2V masking 可代替架構 blind-spot）|
| MSE vs L₁ | 中等（重尾噪聲 L₁ 更穩健）| 輕微（SEM 噪聲接近 Gaussian，MSE 適用）|
| R3 缺失 | 明顯（去噪品質下降）| 中等（仍有改善，但 SEM 噪聲較規則）|

結論：`denoise_apbsn.py` 對 SEM 可產生合理結果，但若需對照文獻 AP-BSN 性能數據，應使用 `faithful.py` 或 `lee.py`。

---

## 8. 結論

### `denoise_apbsn.py`

實質上是「PD 預處理 + N2V 遮蔽訓練 + U-Net」，缺少論文三個核心貢獻（Asymmetric PD、DBSNl、R3）中的兩個半。對 SEM 像素獨立噪聲仍可用，但與論文 AP-BSN 不是同一演算法。

### `denoise_apbsn_faithful.py`

補齊所有論文偏差：DBSNl 架構、L₁ loss、Asymmetric PD、R3。PD 以 numpy 預計算方式實作，R3 每次 pass 包含完整 PD 流程。在論文演算法層面高度一致。

### `denoise_apbsn_lee.py`

最接近官方 GitHub 程式碼的移植版：PD 封裝於 `APBSN.forward()` 內、`pixel_shuffle_down/up_sampling` 與官方 `util.py` 完全一致、官方 R3 邏輯（refinement pass 不走 PD）。剩餘差異為刻意的 SEM 適應（灰階單通道、單張影像訓練、無 data augmentation）。

---

## 參考資料

- Lee et al., [AP-BSN: Self-Supervised Denoising for Real-World Images via Asymmetric PD and Blind-Spot Network](https://arxiv.org/abs/2203.11799), CVPR 2022
- [CVPR 2022 Official Paper (PDF)](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_AP-BSN_Self-Supervised_Denoising_for_Real-World_Images_via_Asymmetric_PD_and_CVPR_2022_paper.pdf)
- [Official GitHub: wooseoklee4/AP-BSN](https://github.com/wooseoklee4/AP-BSN)
- [DBSNl Source Code](https://github.com/wooseoklee4/AP-BSN/blob/master/src/model/DBSNl.py)
- [APBSN Model Wrapper](https://github.com/wooseoklee4/AP-BSN/blob/master/src/model/APBSN.py)

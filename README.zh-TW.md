<p align="center">
  中文 | <a href="README.md">English</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue" />
  <img src="https://img.shields.io/badge/PyTorch-supported-red" />
  <img src="https://img.shields.io/badge/CUDA-supported-green" />
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey" />
  <img src="https://img.shields.io/badge/Video-NVENC-orange" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLO%20%7C%20Florence--2%20%7C%20GroundingDino%20%7C%20SAM--2%20%7C%20LaMa-purple" />
  <img src="https://img.shields.io/badge/WebUI-Gradio-yellow" />
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" />
</p>

# Watermark Remover AI Pro - 支援切片與旋轉偵測的專業級多模型 AI 浮水印移除工具

這是一個利用 AI 模型自動偵測和辨識圖像和影片裡的浮水印並將之去除的工具。本專案可挑選 YOLO、Florence-2、GroundingDino 進行浮水印辨識偵測，可搭配 SAM2 精細畫遮罩範圍，並提供分區掃描及旋轉偵測目標之功能。偵測框產生遮罩檔 Mask 後，再用 LaMa 模型進行高品質圖像修復。各項偵測及去除參數可人工介入調整。

## ✨ 主要功能

- **支援多種媒體**：可處理圖像（`.png`, `.jpg`等）和影片（`.mp4`, `.mov`等）檔案。

- **四種浮水印偵測與處理模型**：
  - **YOLO 模型**：速度快，適用於檢測固定樣式的浮水印。
  - **Florence-2 模型**：靈活性高，透過 LLM Prompt 交談來引導 AI 定位浮水印。
  - **GroundingDino 模型**：專注 detection + grounding，透過 Text 指示 AI 定位浮水印。
  - **SAM2 模型**：將識別出的遮罩方框精細貼近目標物輪廓，而非矩形樣態。

- **高品質修復**：使用 LaMa 模型進行圖像修復（Inpainting），效果自然，痕跡少。

- **雙操作介面**：
  - **WebUI**：提供簡單易用的 Web 圖形化介面，所有參數均有詳細說明，適合大多數使用者。
  - **命令列**：透過 `main.py` 提供完整的命令列介面，方便整合和自動化調用。

- **性能優化**：
  - **批次處理**：對檢測和修復環節都提供批次處理，大幅提升影片處理速度。
  - **GPU加速**：完整支援NVIDIA GPU（CUDA），並為影片編碼提供 NVENC 硬體加速支援。
  - **半精度推理**：支援 FP16 半精度，在相容的GPU上可減少 VRAM 使用並提升速度。

- **加強偵測機制**：
  - **分區及旋轉偵測**：引進多重區域分割檢視及偵測框旋轉功能，以對付隨機散佈及角度傾斜之浮水印。

---

### 範例:

### 圖片+Yolo

![Image Example](example_img.jpg)

### 影片+Florence-2+SAM2

![Video Example](example_vid.jpg)

---

## 🚀 安裝與設定

### 1. 環境要求

- Python 3.12+
- `pip` (Python 套件管理器)
- **FFmpeg**：處理影片音訊的必要工具。請確保你的系統中有安裝，並將其新增到系統環境變數（PATH）中。從 [**FFMpeg**](https://www.ffmpeg.org/download.html) 下載安裝。
- **NVIDIA GPU (推薦)**：為了獲得理想的處理速度，強烈建議使用 NVIDIA 顯示卡，並安裝對應的 [**CUDA Toolkit**](https://developer.nvidia.com/cuda-toolkit) 和 [**驅動程式**](https://www.nvidia.com/drivers/)。

### 2. 安裝步驟

1.  **Clone GitHub 倉庫** (如果還未下載):

    ```bash
    git clone https://github.com/OhMy-Git/WatermarkRemover-AI-Pro.git WatermarkRemover-AI-Pro
    cd WatermarkRemover-AI-Pro
    ```

2.  **安裝 Python 依賴**:
    已經準備好 `requirements.txt` 檔案。執行以下命令以安裝所有必要的函式庫：

    ```bash
    pip install -r requirements.txt
    ```

3.  **安裝 GroundingDino**:

    ```bash
    cd src/grdoundingdino
    pip install ".[dev]"

    或是
    cd src/grdoundingdino
    pip install --no-build-isolation -e .
    ```

4.  **下載模型與權重檔案**:
    本專案需要以下模型檔案，請下載並將它們放置在正確的目錄中：
    - **YOLO 模型** (用於檢測):
      - 從 [這裡](https://huggingface.co/corzent/yolo11x_watermark_detection/blob/main/best.pt) 下載浮水印相關模型然後改名為 `yolo.pt` 。
      - 或將您自己訓練的 YOLO 模型檔案（需命名為 `yolo.pt`）放置在 `models/` 目錄下。
      - 最終路徑應為：`models/yolo.pt`

    - **LaMa 模型** (用於修復):
      - 從 [這裡](https://huggingface.co/fashn-ai/LaMa/resolve/main/big-lama.pt) 下載 `big-lama.pt` 檔案。
      - 將下載的檔案放置在 `models/` 目錄下。
      - 最終路徑應為：`models/big-lama.pt`

    - **Florence-2 模型**
      - 會在首次使用時自動由 `transformers` 函式庫下載，請確保連上 internet。

    - **GoundingDino 和 SAM2 權重及設定檔**
      - 亦請下載後依專案架構所示命名並放置於 weights 目錄下。
      - 從 [這裡](https://huggingface.co/pengxian/grounding-dino/tree/main) 下載 `groundingdino` 檔案。
      - 從 [這裡](https://huggingface.co/facebook/sam2-hiera-large/tree/main) 下載 `sam2_hiera_large.pt` 及 `sam2_hiera_l.yaml` 檔案。

---

## 📁 專案架構

```
WatermarkRemover-AI/
├── main.py                    # 主程式，命令列入口
├── webui.py                   # WebUI 介面
├── check_env.py               # 環境配置檢查程式
├── requirements.txt           # Python 依賴
├── src/groundingdino          # GroundingDino setup
├── models/                    # 模型檔案目錄
│   ├── yolo.pt                # YOLO 檢測模型（需手動下載）
│   └── big-lama.pt            # LaMa 修復模型（需手動下載）
├── weights/                   # 模型權重目錄
│   ├── groundingdino_swinb_cfg.py
│   ├── groundingdino_swinb_cogcoor.pth
│   ├── sam2_hiera_b+.yaml
│   ├── sam2_hiera_l.yaml
│   └── sam2_hiera_large.pt
└── output_webui/              # WebUI 輸出目錄


```

## 🕹️ 使用方式

你可以透過兩種方式使用本工具：

### 1. WebUI (推薦)

這是最簡單直觀的使用方式。在專案根目錄下執行：

```bash
python webui.py
```

程式會啟動一個本地 Web 服務（通常地址為 `http://127.0.0.1:7860`）。在瀏覽器中開啟此地址即可看到操作介面。所有參數在介面上都有解釋。

### 2. 命令列 (`main.py`)

對於進階使用者或需要批次自動化的場景，可以直接使用 `main.py`。

**基本用法:**

```bash
python main.py --input <輸入檔案或目錄> --output <輸出目錄> [其他選項]
```

### 4. 主要命令列參數詳解:

| 參數                          | 預設值  | 說明                                           |
| :---------------------------- | :------ | :--------------------------------------------- |
| `--resize-limit`              | `768`   | 影像送入修復模型前的最大尺寸。                 |
| `--BBox Expand Pixels`        | `0    ` | 偵測框外擴像素。                               |
| `--Max BBox Area (%)`         | `10`    | 浮水印偵測框佔影像總面積的最大百分比。         |
| `--lama-scale`                | `1`     | LaMa 重繪解析度的縮放因子。                    |
| `--Rotation Step (°)`         | `0`     | 偵測框旋轉以偵測傾斜浮水印的步進角度數。       |
| `--tiles-per-axis`            | `3`     | 分區偵測的每邊切格數量。總分區數為數值的平方。 |
| `--Tile Overlap`              | `0.25`  | 分區偵測時各分區格之間的重疊比例。             |
| `--Use FP16`                  | 啟用    | 是否啟用FP16半精度推理。                       |
| `--Use SAM-2 Mask Refine`     | 停用    | 是否使用 SAM2 精細化遮罩輪廓。                 |
| `--CLAHE Mode`                | off     | 增強圖像對比突出浮水印以利偵測。               |
| `--video-batch-size`          | `8`     | 處理影片時的批次大小，影響速度和顯存佔用。     |
| `--use-first-frame-detection` | 禁用    | 是否對影片啟用「首幀檢測」模式。               |

**YOLO專用參數:**

| 參數                 | 預設值 | 說明                            |
| :------------------- | :----- | :------------------------------ |
| `--conf-threshold`   | `0.6`  | YOLO 檢測的置信度閾值。         |
| `--iou-threshold`    | `0.45` | YOLO 的 IOU 閾值。              |
| `--Single Detection` | 啟用   | 使用快速辨識模式                |
| `--Yolo Image Size`  | `640`  | YOLO 模型推理時使用的圖像尺寸。 |

---

## ⚡ 性能優化

- **使用 GPU**：這是最重要的性能保障。請確保你的環境已正確配置 CUDA。若無則會自動以 CPU 運行，速度很慢。
- **首幀檢測**：處理浮水印位置固定的影片時，務必在WebUI中勾選「Use First Frame Mask for Videos」，或在命令列使用 `--use-first-frame-detection`。這是**最有效**的影片處理加速手段。
- **調整批次大小 (`Video Batch Size`)**：根據你的 GPU VRAM 大小，適當增大此參數可以提升 GPU 利用率，加快影片處理速度。如果遇到 VRAM 不足（Out of Memory）的錯誤，請減小此值。
- **調整YOLO圖像尺寸 (`YOLO Image Size`)**：減小此值（如從1280降到640）可以顯著加快 YOLO 的檢測速度，但可能會犧牲對微小水印的檢測精度。你可以根據實際情況進行權衡調整。

---

## 🛠 開發工具

- 使用 Python 3.12 語法
- 遵循 PEP 8 程式碼規範
- 使用 loguru 進行日誌記錄
- 使用 click 進行命令列參數解析
- 使用 PyTorch 進行深度學習模型推理

---

## 🧠 技術指引

- 深度學習框架：PyTorch
- 計算機視覺：OpenCV
- Web 框架：Gradio（WebUI）
- 模型庫：
  - ultralytics（YOLO）v11 for watermark
  - transformers（Florence-2）
  - LaMa（遮罩區重繪）
- 工具庫：numpy, PIL, tqdm, loguru

---

## ❓ 常見問題

1. **模型檔案缺失**：確保已下載模型檔案並以正確檔名放置於 models 資料夾
2. **CUDA 無法使用**：檢查 NVIDIA 顯卡驅動程式和 CUDA Toolkit 安裝狀態:檢查 torch.cuda.is_available()
3. **FFmpeg 錯誤**：確保系統已安裝 FFmpeg 並新增到系統 PATH ，驗證於作業系統中可供呼叫使用: FFmpeg
4. **顯卡記憶體(VRAM、顯存) 不足**：減小批次處理大小或禁用半精度推理

---

## 💡 建議

1. 若輸出結果還是有浮水印去除不乾淨的地方，可使用手動標示去除工具 Lama Cleaner 或 IOPaint 等專案進行手動後製精細處理。
2. 浮水印位置固定之影片，可先擷取第一幀存成影像檔，先將其以影像模式調整參數運行後，看其產出之遮罩檔 mask 偵測位置是否正確，
   再套用正確參數到影片模式執行。
3. 隨時關注 Yolo 模型之相關進度，取用適合版本。

本專案受到 https://comfyui.org/en/ai-powered-watermark-removal-workflow 啟發。

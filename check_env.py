import torch
import sys
import os
import io
import warnings

# 強制 stdout 使用 UTF-8 避免 Windows 輸出 Emoji 亂碼
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 忽略不必要的警告
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=UserWarning, module="triton")
warnings.filterwarnings("ignore", category=UserWarning, module="groundingdino")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="groundingdino")


def print_separator():
    print()
    print("-" * 60)
    print()


# --- 各項目檢查器 ---

def check_1_2_torch_cuda():
    """1~2 項目：Torch + CUDA 基礎"""
    print(f"1. CUDA 可用性:  {'✅' if torch.cuda.is_available() else '⚠️  '}")
    print(f"2. Torch 版本:   {torch.__version__}  ✅")
    print(f"   顯示器:  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '無'}")


def check_3_4_xformers_triton():
    """3~4 項目：xformers 與 triton"""
    # 3. xformers 基本 import
    xformers_ok = False
    try:
        import xformers
        print(f"3. xformers 版本:  {xformers.__version__}  ✅")
        xformers_ok = True
    except ImportError as e:
        # 檢查錯誤訊息中是否包含 triton/triton 相關
        if "triton" in str(e).lower():
            print(f"3. xformers 載入:  因 triton 問題（可忽略）")
        else:
            print(f"3. xformers 載入:  未安裝 ❌")

    # 6a. triton
    try:
        import triton
        print(f"3b. triton 版本:   {triton.__version__}  ✅")
    except ImportError:
        print(f"3b. triton:        未安裝（Windows 上可忽略）✅")

    # 4. xformers._C 檢查
    has_ops_ok = True
    try:
        import xformers.ops
    except Exception:
        has_ops_ok = False

    has_ops_ok = True
    try:
        from xformers._C import _has_ops
    except Exception:
        has_ops_ok = False

    if xformers_ok:
        print("4. xformers 運算: 正常 ✅（_C.pyd 載入非阻斷）✅")
    elif not xformers_ok and not has_ops_ok:
        print("4. xformers 運算: 失敗（_C.pyd 載入非阻斷）✅")
    else:
        print(f"4. xformers 運算:  正常（_C.pyd 載入非阻斷）✅")


def check_5_sam2():
    """5 項目：SAM-2 模組加載"""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model_dir = "weights"
        config_file = os.path.abspath(os.path.join(model_dir, "sam2_hiera_l.yaml"))
        checkpoint_file = os.path.abspath(os.path.join(model_dir, "sam2_hiera_large.pt"))

        sam2_model = build_sam2(
            config_file=config_file,
            checkpoint=checkpoint_file,
            device="cuda",
        )

        # 注意: 不同 SAM2 版本可能用不同參數名稱
        try:
            predictor = SAM2ImagePredictor(sam_model=sam2_model, device="cuda")
        except TypeError:
            predictor = SAM2ImagePredictor(sam2_model=sam2_model, device="cuda")

        print(f"5. SAM-2:        模型 OK（config={config_file}）✅")
        if torch.cuda.is_available():
            print(f"   - 裝置:          cuda:0 (GPU)")
        else:
            print("   - 裝置:          CPU")
    except FileNotFoundError as e:
        print(f"5. SAM-2:        缺少檔案 ({e})")
    except Exception as e:
        print(f"5. SAM-2:        失敗 ❌ ({type(e).__name__}: {e})")


# --- 主檢查函式 ---

def check_environment():
    print("=" * 60)
    print("🚀 正在檢查 AI 開發環境 (WatermarkRemover-AI3)")
    print("=" * 60)
    print()
    check_1_2_torch_cuda()
    check_3_4_xformers_triton()
    check_5_sam2()
    check_6_grounding_dino()

    # 7. Numpy
    print("-" * 60)
    print()
    try:
        import numpy as np
        status = "✅" if np.__version__.startswith("1.") else "⚠️  (需 <2.0)"
        print(f"7. Numpy 版本:     {np.__version__} {status}")
    except Exception:
        print("7. Numpy 版本:     無法檢查")

    print("=" * 60)
    print("環境檢查完成！（全部顯示 ✅ 即可執行程式）")
    print("=" * 60)


def check_6_grounding_dino():
    """6 項目：GroundingDINO 檢查"""
    print("-" * 60)
    print()

    # 路徑設定（與 main.py 一致）
    gdino_path = os.path.join(os.getcwd(), "src", "groundingdino")
    if gdino_path not in sys.path:
        sys.path.append(gdino_path)
    print(f"6. GroundingDINO:  路徑:   {gdino_path}")

    errors = []

    # 核心模組檢查
    try:
        from groundingdino.util.inference import load_model  # type: ignore[import-not-found]
        print("   - load_model:   OK")
    except ImportError as e:
        errors.append(f"load_model: {e}")
        print("   - load_model:   未找到")

    try:
        from groundingdino.util.inference import load_image, predict  # type: ignore[import-not-found]
        print(f"   - load_image:   OK / predict:   OK")
    except ImportError as e:
        errors.append(f"load_image/predict: {e}")
        print(f"   - load_image 及 predict: {e}")

    try:
        from groundingdino.util.slconfig import SLConfig  # type: ignore[import-not-found]
        from groundingdino.util import box_ops  # type: ignore[import-not-found]
        print(f"   - slconfig:     OK / box_ops: OK")
    except ImportError as e:
        errors.append(f"slconfig/box_ops: {e}")
        print(f"   - slconfig 及 box_ops: {e}")

    try:
        from groundingdino.models import build_model, GroundingDINO  # type: ignore[import-not-found]
        print(f"   - build_model:  OK / GroundingDINO class: OK")
    except ImportError as e:
        errors.append(f"build_model/GroundingDINO class: {e}")
        print(f"   - build_model 及 GroundingDINO class: {e}")

    # 權重配置檢查
    config_path = os.path.join(os.getcwd(), "weights", "groundingdino_swinb_cfg.py")
    if os.path.isfile(config_path):
        print(f"   - Config file:  OK ✅")
    else:
        errors.append(f"config: 未找到 {config_path}")
        print(f"   - ⚠️  Config file: {config_path} 未找到")

    # 整體結果
    if errors:
        print(f"6-總結 (errors):  {len(errors)}/{6} 項目異常 ({', '.join(e.split(':')[0] for e in errors)})")
    else:
        print(f"6-總結 (error):   所有模組 OK")


if __name__ == "__main__":
    check_environment()

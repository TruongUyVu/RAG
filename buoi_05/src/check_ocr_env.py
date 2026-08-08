"""
=============================================================================
BUỔI 05: KIỂM TRA MÔI TRƯỜNG PYTHON & CÁC THƯ VIỆN XỬ LÝ VĂN BẢN / OCR
=============================================================================
Script kiểm tra tính sẵn sàng của các công cụ:
- Python Version
- PyMuPDF (fitz)
- Pillow (PIL)
- Llama_cloud (llama_cloud)
- Pydantic (pydantic)
- Streamlit (streamlit)
- python-dotenv (dotenv)
=============================================================================
"""

import sys
import io
import subprocess
import importlib.util
from typing import Dict, Tuple

# Đảm bảo in tiếng Việt chuẩn mã hóa UTF-8 trên Windows Console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Danh sách các thư viện cần kiểm tra: (Tên hiển thị, Tên module import, Package pip tương ứng)
REQUIRED_PACKAGES = [
    ("Python Version", "python", "python"),
    ("PyMuPDF", "fitz", "pymupdf"),
    ("Pillow", "PIL", "pillow"),
    ("Llama_cloud", "llama_cloud", "llama-cloud"),
    ("Pydantic", "pydantic", "pydantic"),
    ("Streamlit", "streamlit", "streamlit"),
    ("python-dotenv", "dotenv", "python-dotenv"),
]

def check_package(display_name: str, module_name: str) -> Tuple[bool, str]:
    """Kiểm tra xem module đã được cài đặt và import thành công chưa."""
    if module_name == "python":
        version_str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info >= (3, 8):
            return True, f"Python {version_str}"
        return False, f"Python {version_str} (Yêu cầu >= 3.8)"

    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "Đã cài đặt")
            return True, f"v{version}"
        else:
            return False, "Chưa cài đặt (Missing)"
    except Exception as e:
        return False, f"Lỗi import: {str(e)}"

def run_environment_check() -> Dict[str, Dict[str, str]]:
    """Chạy kiểm tra toàn bộ danh sách và in bảng kết quả PASS/FAIL."""
    results = {}
    
    print("\n" + "=" * 70)
    print("   BẢNG KIỂM TRA MÔI TRƯỜNG PYTHON & THƯ VIỆN (OCR / RAG)")
    print("=" * 70)
    print(f"| {'STT':<4} | {'Tên Công Cụ':<18} | {'Trạng Thái':<10} | {'Chi Tiết / Phiên Bản':<26} |")
    print("|" + "-"*6 + "|" + "-"*20 + "|" + "-"*12 + "|" + "-"*28 + "|")

    all_pass = True
    for idx, (display_name, module_name, pip_name) in enumerate(REQUIRED_PACKAGES, 1):
        is_pass, info = check_package(display_name, module_name)
        status_str = "PASS ✅" if is_pass else "FAIL ❌"
        if not is_pass:
            all_pass = False

        print(f"| {idx:<4} | {display_name:<18} | {status_str:<10} | {info:<26} |")
        
        results[module_name] = {
            "display_name": display_name,
            "pip_name": pip_name,
            "status": "PASS" if is_pass else "FAIL",
            "info": info
        }

    print("=" * 70)
    if all_pass:
        print("🎉 TẤT CẢ CÔNG CỤ ĐÃ SẴN SÀNG! Môi trường hoạt động hoàn hảo.\n")
    else:
        print("⚠️ CÓ CÔNG CỤ CHƯA SẴN SÀNG! Cần tiến hành cài đặt khắc phục.\n")

    return results

def auto_fix_failed_packages(results: Dict[str, Dict[str, str]]):
    """Tự động cài đặt các thư viện bị FAIL."""
    failed_pips = []
    for mod, data in results.items():
        if data["status"] == "FAIL" and data["pip_name"] != "python":
            failed_pips.append(data["pip_name"])

    if not failed_pips:
        print("✅ Không có gói thư viện nào bị lỗi cần khắc phục.")
        return

    print(f"\n🛠️ Bắt đầu tự động khắc phục cho các gói: {', '.join(failed_pips)}")
    cmd = [sys.executable, "-m", "pip", "install"] + failed_pips
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Cài đặt thành công tất cả các thư viện bị thiếu!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Cài đặt thất bại. Lỗi: {e.stderr}")

if __name__ == "__main__":
    res = run_environment_check()
    if "--fix" in sys.argv:
        auto_fix_failed_packages(res)
        print("\n--- KIỂM TRA LẠI SAU KHI KHẮC PHỤC ---")
        run_environment_check()

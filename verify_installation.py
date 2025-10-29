"""
Installation Verification Script for Ditto TalkingHead

This script verifies that all required dependencies are correctly installed.
"""

import sys
import os

def check_python_version():
    """Check Python version is 3.10.x"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor == 10:
        print(f"  ✓ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python version: {version.major}.{version.minor}.{version.micro}")
        print(f"    Required: Python 3.10.x")
        return False

def check_torch():
    """Check PyTorch installation and CUDA availability"""
    print("\nChecking PyTorch...")
    try:
        import torch
        print(f"  ✓ PyTorch installed: {torch.__version__}")

        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: True")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            print(f"  ✓ GPU device: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ Number of GPUs: {torch.cuda.device_count()}")
        else:
            print(f"  ✗ CUDA available: False")
            print(f"    Warning: CUDA not available. GPU acceleration will not work.")
            print(f"    Check: nvidia-smi and CUDA installation")
            return False

        return True
    except ImportError as e:
        print(f"  ✗ PyTorch not installed: {e}")
        return False

def check_torchvision():
    """Check torchvision installation"""
    print("\nChecking torchvision...")
    try:
        import torchvision
        print(f"  ✓ torchvision installed: {torchvision.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ torchvision not installed: {e}")
        return False

def check_torchaudio():
    """Check torchaudio installation"""
    print("\nChecking torchaudio...")
    try:
        import torchaudio
        print(f"  ✓ torchaudio installed: {torchaudio.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ torchaudio not installed: {e}")
        return False

def check_numpy():
    """Check numpy installation"""
    print("\nChecking numpy...")
    try:
        import numpy as np
        print(f"  ✓ numpy installed: {np.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ numpy not installed: {e}")
        return False

def check_librosa():
    """Check librosa installation"""
    print("\nChecking librosa (audio processing)...")
    try:
        import librosa
        print(f"  ✓ librosa installed: {librosa.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ librosa not installed: {e}")
        return False

def check_opencv():
    """Check OpenCV installation"""
    print("\nChecking OpenCV (video processing)...")
    try:
        import cv2
        print(f"  ✓ OpenCV installed: {cv2.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ OpenCV not installed: {e}")
        return False

def check_scipy():
    """Check scipy installation"""
    print("\nChecking scipy...")
    try:
        import scipy
        print(f"  ✓ scipy installed: {scipy.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ scipy not installed: {e}")
        return False

def check_tensorrt():
    """Check TensorRT installation (optional but recommended)"""
    print("\nChecking TensorRT (optional)...")
    try:
        import tensorrt
        print(f"  ✓ TensorRT installed: {tensorrt.__version__}")
        return True
    except ImportError as e:
        print(f"  ! TensorRT not installed (optional)")
        print(f"    TensorRT provides 2-3x speedup but is optional")
        return None  # None = optional dependency

def check_imageio():
    """Check imageio installation"""
    print("\nChecking imageio...")
    try:
        import imageio
        print(f"  ✓ imageio installed: {imageio.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ imageio not installed: {e}")
        return False

def check_sklearn():
    """Check scikit-learn installation"""
    print("\nChecking scikit-learn...")
    try:
        import sklearn
        print(f"  ✓ scikit-learn installed: {sklearn.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ scikit-learn not installed: {e}")
        return False

def check_soundfile():
    """Check soundfile installation"""
    print("\nChecking soundfile...")
    try:
        import soundfile
        print(f"  ✓ soundfile installed: {soundfile.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ soundfile not installed: {e}")
        print(f"    Hint: Install system library: sudo apt-get install libsndfile1")
        return False

def check_pillow():
    """Check Pillow installation"""
    print("\nChecking Pillow (image processing)...")
    try:
        import PIL
        print(f"  ✓ Pillow installed: {PIL.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ Pillow not installed: {e}")
        return False

def check_yaml():
    """Check PyYAML installation"""
    print("\nChecking PyYAML...")
    try:
        import yaml
        print(f"  ✓ PyYAML installed")
        return True
    except ImportError as e:
        print(f"  ✗ PyYAML not installed: {e}")
        return False

def check_ffmpeg():
    """Check ffmpeg installation (system dependency)"""
    print("\nChecking ffmpeg (system dependency)...")
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                              capture_output=True,
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"  ✓ ffmpeg installed: {version_line}")
            return True
        else:
            print(f"  ✗ ffmpeg not found")
            print(f"    Install: sudo apt-get install ffmpeg")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"  ✗ ffmpeg not found")
        print(f"    Install: sudo apt-get install ffmpeg")
        return False

def check_cuda_system():
    """Check CUDA system installation"""
    print("\nChecking CUDA system installation...")
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'],
                              capture_output=True,
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            # Extract CUDA version from nvidia-smi output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    print(f"  ✓ NVIDIA Driver installed")
                    print(f"  ✓ CUDA Version: {cuda_version}")
                    return True
            print(f"  ✓ NVIDIA Driver installed")
            return True
        else:
            print(f"  ✗ nvidia-smi failed")
            print(f"    Check: NVIDIA driver installation")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"  ✗ nvidia-smi not found")
        print(f"    Check: NVIDIA driver and CUDA installation")
        return False

def check_project_structure():
    """Check if project structure is correct"""
    print("\nChecking project structure...")
    required_files = [
        'inference.py',
        'inference_video.py',
        'inference_video_advanced.py',
        'stream_pipeline_offline.py',
        'core/atomic_components/putback_configurable.py',
    ]

    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} not found")
            all_exist = False

    return all_exist

def main():
    """Run all verification checks"""
    print("=" * 70)
    print("Ditto TalkingHead - Installation Verification")
    print("=" * 70)

    checks = [
        ("Python Version", check_python_version),
        ("PyTorch", check_torch),
        ("torchvision", check_torchvision),
        ("torchaudio", check_torchaudio),
        ("numpy", check_numpy),
        ("librosa", check_librosa),
        ("OpenCV", check_opencv),
        ("scipy", check_scipy),
        ("imageio", check_imageio),
        ("scikit-learn", check_sklearn),
        ("soundfile", check_soundfile),
        ("Pillow", check_pillow),
        ("PyYAML", check_yaml),
        ("CUDA System", check_cuda_system),
        ("ffmpeg", check_ffmpeg),
        ("TensorRT", check_tensorrt),
        ("Project Structure", check_project_structure),
    ]

    results = []
    optional_failed = []

    for name, check_func in checks:
        try:
            result = check_func()
            if result is None:
                optional_failed.append(name)
            else:
                results.append((name, result))
        except Exception as e:
            print(f"  ✗ Error checking {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    failed = sum(1 for _, result in results if not result)
    total = len(results)

    print(f"\nRequired checks:")
    print(f"  ✓ Passed: {passed}/{total}")
    if failed > 0:
        print(f"  ✗ Failed: {failed}/{total}")
        print("\nFailed checks:")
        for name, result in results:
            if not result:
                print(f"    - {name}")

    if optional_failed:
        print(f"\nOptional checks (not installed):")
        for name in optional_failed:
            print(f"    - {name}")

    print("\n" + "=" * 70)

    if failed == 0:
        print("✓ ALL REQUIRED CHECKS PASSED!")
        print("✓ Your environment is ready to use.")
        print("\nNext steps:")
        print("  1. Download model checkpoints")
        print("  2. Run: python inference_video.py --help")
        print("  3. Read: personal_docs/README.md")
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nPlease fix the failed checks and run this script again.")
        print("See SETUP_UV.md for troubleshooting help.")
        return 1

    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())

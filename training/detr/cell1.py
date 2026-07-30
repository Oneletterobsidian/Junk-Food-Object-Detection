import subprocess, sys, os

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "torch==2.2.0+cu118", "torchvision==0.17.0+cu118",
    "--index-url", "https://download.pytorch.org/whl/cu118", "-q"
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "numpy==1.26.4", "--force-reinstall", "-q"
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "transformers==4.40.0", "pycocotools", "-q"
])
subprocess.check_call([
    sys.executable, "-m", "pip", "uninstall", "ray", "-y"
])

os.execv(sys.executable, [sys.executable] + sys.argv)
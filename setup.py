from setuptools import setup, find_packages
import os
import subprocess
import platform

def is_linux():
    return platform.system() == 'Linux'

if is_linux():
    print("The system is running on Linux.")
else:
    print("The system is not running on Linux.")

def install_dependencies():
    torch_dependencies = "https://download.pytorch.org/whl/torch_stable.html"

    if is_linux == 0:
        subprocess.check_call(["pip", "install", "torch==2.0.0+cu118", "torchvision==0.15.1+cu118", "-f", torch_dependencies])
    else:
        subprocess.check_call(["pip", "install", "torch==2.0.0", "torchvision==0.15.1", "-f", torch_dependencies])
        
    subprocess.check_call(["pip", "install", "SciencePlots"])
    subprocess.check_call(["pip", "install", "openmim"])
    subprocess.check_call(["mim", "install", "mmengine"])
    subprocess.check_call(["mim", "install", "mmcv>=2.0.0"])
    subprocess.check_call(["pip", "install", "mmdet"])
    # subprocess.check_call(["git", "clone", "https://github.com/open-mmlab/mmdetection.git"])
    # os.chdir("./mmdetection")
    # subprocess.check_call(["pip", "install", "-v", "-e", "."])
    # os.chdir("..")
    # subprocess.check_call(["pip", "install", "tifffile"])

    os.remove("mmdet.egg-info")

install_dependencies()

setup(
    name="navi_project",
    version="0.1",
    packages=find_packages(),
)

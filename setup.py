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
    mmcv_dependencies = "https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html"

    if is_linux:
       subprocess.check_call(["pip", "install", "torch==1.9.0+cu111", "torchvision==0.10.0+cu111", "-f", torch_dependencies])
    else:
       subprocess.check_call(["pip", "install", "torch==1.9.0", "torchvision==0.10.0", "-f", torch_dependencies])
    subprocess.check_call(["pip", "install", "mmcv-full", "-f", mmcv_dependencies])

    subprocess.check_call(["rm", "-rf", "mmdetection"])
    subprocess.check_call(["git", "clone", "https://github.com/open-mmlab/mmdetection.git"])
    os.chdir("mmdetection")
    subprocess.check_call(["pip", "install", "-e", "."])
    os.remove("mmdet.egg-info")

install_dependencies()

setup(
    name="navi_project",
    version="0.1",
    packages=find_packages(),
)

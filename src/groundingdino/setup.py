# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -- ------------------------- --- ------ ----- -------------------  --- ------
# Modified from
# https://github.com/fundamentalvision/deformable-detr/blob/main/models/ops/setup.py
# https://github.com/facebookresearch/detectron2/blob/main/setup.py
# https://github.com/open-mmlab/mmdetection/blob/master/setup.py
# https://github.com/Oneflow-Inc/libai/blob/main/setup.py
#
# ------ ------ ------ ----- ----------- ---------  -----  ------ ---------- ---- ------ -----

import glob
import os
import subprocess
import sys
import re
from io import StringIO
from os.path import exists
from setuptools import find_packages, setup


def _try_import_torch():
    """Import torch.  If not available, install it (once).
    Returns True if torch is available.
    """
    # Check sys.modules + already installed
    if "torch" in sys.modules:
        return True

    try:
        __import__("torch")
        return True
    except ImportError:
        pass  # will install

    # Install via a fresh Python process using sys.executable
    # This calls `python -m pip install torch torchvision -q`
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0 or result.returncode == 1:  # 1 can be OK
            __import__("torch")
            return True
        # If pip was the issue, try using pip directly
        pass
    except Exception:
        pass

    # Try using pip installed in the same directory as our python.exe
    try:
        pip_path = os.path.join(
            os.path.dirname(sys.executable),
            "Scripts", "pip.exe",
        )
        if os.path.isfile(pip_path):
            result = subprocess.run(
                [pip_path, "-q", "install", "torch", "torchvision"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode == 0 or result.returncode == 1:
                __import__("torch")
                return True
    except Exception:
        pass

    # Final fallback: try to import without forcing install
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# -- version info ------ --------------------  -------------- ---------- -----
version = "0.1.0"
package_name = "groundingdino"
cwd = os.path.dirname(os.path.abspath(__file__))

_sha = "Unknown"
try:
    _sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode(
        "ascii", "replace",
    ).strip()
except Exception:
    pass


def write_version_file():
    vpath = os.path.join(cwd, "groundingdino", "version.py")
    with open(vpath, "w") as f:
        f.write(f"__version__ = '{version}'\n")


# -- requirements parse ------ ------- ----------------  ---  -----------  --- -- -
def parse_requirements(fname="requirements.txt", with_version=True):
    req = fname
    if not os.path.isfile(req):
        return []

    def parse_line(line):
        if line.startswith("-r "):
            for info in parse_require_file(line.split(" ", 1)[1]):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            elif "@git+" in line:
                info["package"] = line
            else:
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                pts = re.split(pat, line, maxsplit=2)
                pts = [p.strip() for p in pts]
                info["package"] = pts[0]
                if len(pts) > 1:
                    op = pts[1]
                    rest = pts[2] if len(pts) > 2 else ""
                    if ";" in rest:
                        vsn, plat = map(str.strip, rest.split(";"))
                        info["platform_deps"] = plat
                        info["version"] = (op, vsn)
                    else:
                        info["version"] = (op, rest)
            yield info

    def parse_require_file(fpath):
        with open(fpath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    for info in parse_line(line):
                        yield info

    def gen():
        for info in parse_require_file(req):
            parts = [info["package"]]
            if with_version and "version" in info:
                parts.append(info["version"][0])
                parts.append(info["version"][1])
            if not sys.version.startswith("3.4"):
                plat = info.get("platform_deps")
                if plat is not None:
                    parts.append(";" + plat)
            yield "".join(parts)

    return list(gen())


# -- extensions ------ -----------------  ----------   ---------------  ---------
def _try_import_extensions():
    """Import torch and return build extensions module (or None if absent)."""
    try:
        import torch
        import torch.utils.cpp_extension
        return {
            "cpp_ext": torch.utils.cpp_extension.CppExtension,
            "cuda_ext": torch.utils.cpp_extension.CUDAExtension,
            "cuda_home": torch.utils.cpp_extension.CUDA_HOME,
            "is_cuda": torch.cuda.is_available(),
            "build_ext": torch.utils.cpp_extension.BuildExtension,
        }
    except (ImportError, AttributeError, OSError):
        return None


def _get_extensions():
    """Build cuda or cpu extension modules.
    Returns (ext_modules, build_ext_class).
    """
    ext_info = _try_import_extensions()
    if ext_info is None:
        return None, None

    this_dir = os.path.dirname(os.path.abspath(__file__))
    ext_dir = os.path.join(
        this_dir, "groundingdino", "models", "GroundingDINO", "csrc",
    )

    main = os.path.join(ext_dir, "vision.cpp")
    cpp_sources = [os.path.join(ext_dir, s) for s in
                   glob.glob(os.path.join(ext_dir, "**", "*.cpp")) +
                   [os.path.basename(main)]]
    cuda_sources = [os.path.join(ext_dir, s) for s in
                    glob.glob(os.path.join(ext_dir, "**", "*.cu")) +
                    glob.glob(os.path.join(ext_dir, "*.cu"))]
    all_sources = [os.path.join(ext_dir, s) for s in [os.path.basename(main)] +
                   glob.glob(os.path.join(ext_dir, "**", "*.cpp")) +
                   glob.glob(os.path.join(ext_dir, "**", "*.cu")) +
                   glob.glob(os.path.join(ext_dir, "*.cu"))]

    ext_cls = ext_info["cpp_ext"]
    extra_compile_args = {"cxx": []}
    define_macros = []

    if ext_info["cuda_home"] is not None and (
        ext_info["is_cuda"] or "TORCH_CUDA_ARCH_LIST" in os.environ
    ):
        print("Compiling with CUDA")
        ext_cls = ext_info["cuda_ext"]
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        print("Compiling without CUDA")
        define_macros += [("WITH_HIP", None)]
        extra_compile_args["nvcc"] = []

    ext_modules = [
        ext_cls(
            "groundingdino._C",
            all_sources,
            include_dirs=[ext_dir],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules, ext_info["build_ext"]


# -- main ------------  -------   -------     -----------------  ------
def _main():
    print("Building wheel {}-{}".format(package_name, version))

    with open("LICENSE", "r", encoding="utf-8") as f:
        license_ = f.read()

    # Write version file
    write_version_file()

    try:
        _try_import_torch()
    except Exception:
        pass  # install may or may not succeed

    ext_modules, build_ext_cls = _get_extensions()


    cmdclass = {}
    if build_ext_cls is not None:
        cmdclass["build_ext"] = build_ext_cls

    setup(
        name=package_name,
        version=version,
        author="International Digital Economy Academy, Shilong Liu",
        url="https://github.com/IDEA-Research/GroundingDINO",
        description="open-set object detector",
        license=license_,
        install_requires=parse_requirements("requirements.txt"),
        packages=find_packages(exclude=("configs", "tests")),
        ext_modules=ext_modules,
        cmdclass=cmdclass,
    )


# -----------------------------------------------------------------------
# During PEP-517 builds, setup.py is exec'd inside subprocesses.
# The `__name__` becomes `"__main__"`, but _get_extensions() also
# needs torch.  So we ensure torch is available via _try_import_torch().
# -----------------------------------------------------------------------
if __name__ == "__main__":
    _main()

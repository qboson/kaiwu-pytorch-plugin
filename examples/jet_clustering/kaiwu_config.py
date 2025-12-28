# -*- coding: utf-8 -*-
"""
Kaiwu SDK 授权配置脚本
在使用Kaiwu SDK之前，需要先配置授权信息
"""

import os

# 从SDK.md文件读取授权信息
USER_ID = "127342197535145986"
SDK_CODE = "aFFeFH2LtKrrSycGPnp0969awhnUeG"

def init_kaiwu_license():
    """
    初始化Kaiwu SDK授权信息
    使用方法：
        from kaiwu_config import init_kaiwu_license
        import kaiwu as kw
        init_kaiwu_license()  # 会自动调用 kw.license.init()
    """
    try:
        import kaiwu as kw
        kw.license.init(USER_ID, SDK_CODE)
        print(f"Kaiwu SDK授权信息已配置: User ID = {USER_ID}")
        return True
    except ImportError:
        print("警告: kaiwu SDK未安装，请先安装Kaiwu SDK")
        print("请访问 https://platform.qboson.com/sdkDownload 下载并安装")
        return False
    except Exception as e:
        print(f"配置授权信息时出错: {e}")
        return False

def set_environment_variables():
    """
    设置环境变量（可选，用于通过环境变量方式配置）
    """
    os.environ["USER_ID"] = USER_ID
    os.environ["SDK_CODE"] = SDK_CODE
    print(f"环境变量已设置: USER_ID={USER_ID}, SDK_CODE={SDK_CODE[:10]}...")

if __name__ == "__main__":
    # 设置环境变量
    set_environment_variables()
    # 尝试初始化授权
    init_kaiwu_license()


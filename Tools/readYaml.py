import yaml
def load_config(config_path):
    """
    加载 YAML 配置文件并返回配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"配置文件 '{config_path}' 加载成功。")
        return config
    except Exception as e:
        raise FileNotFoundError(f"配置文件加载失败: {e}")
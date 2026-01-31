from src.config.loader import ConfigLoader

cfg = ConfigLoader().load_all()

print(cfg)
print(cfg["base"]["runtime"]["random_state"])
print(cfg["paths"]["data"]["raw"])

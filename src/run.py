import sys

from src.config.loader import ConfigLoader
from src.pipelines.train import train_pipeline
from src.pipelines.app import run_app

cfg = ConfigLoader().load_all()
train_pipeline(cfg)

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.run [train|app]")
        sys.exit(1)

    command = sys.argv[1]

    cfg = ConfigLoader().load_all()

    if command == "train":
        train_pipeline(cfg)

    elif command == "app":
        run_app(cfg)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()

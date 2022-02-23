import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
from config import parse_config

from example.trainer import Trainer


def main() -> None:
    config = parse_config()
    Trainer(config=config).train_and_test_model()


if __name__ == "__main__":
    main()

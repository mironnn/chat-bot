import sys

from dotenv import load_dotenv
from loguru import logger  # type: ignore

load_dotenv()

logger.remove(0)
logger.add(sys.stdout, level="DEBUG", serialize=False)

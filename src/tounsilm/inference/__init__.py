"""TounsiLM inference package."""

from tounsilm.inference.generator import TextGenerator
from tounsilm.inference.api import create_app

__all__ = ["TextGenerator", "create_app"]

import importlib

__modules__ = {}


def register(name):
    def decorator(cls):
        # Allow re-registration for checkpoint loading compatibility
        # When torch.load triggers module re-import, the same class may be registered again
        __modules__[name] = cls
        return cls

    return decorator


def find(name):
    if name in __modules__:
        return __modules__[name]
    else:
        try:
            module_string = ".".join(name.split(".")[:-1])
            cls_name = name.split(".")[-1]
            module = importlib.import_module(module_string, package=None)
            return getattr(module, cls_name)
        except Exception as e:
            raise ValueError(f"Module {name} not found!")


###  grammar sugar for logging utilities  ###
import logging

logger = logging.getLogger("pixal3d")


def debug(*args, **kwargs):
    logger.debug(*args, **kwargs)


def info(*args, **kwargs):
    logger.info(*args, **kwargs)


def warn(*args, **kwargs):
    logger.warning(*args, **kwargs)

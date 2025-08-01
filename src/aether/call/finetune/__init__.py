__IS_PEFT_AVAILABLE__ = False

try:
    import peft

    __IS_PEFT_AVAILABLE__ = True
except ImportError:
    pass

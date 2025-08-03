__IS_ULTRALYTICS_INSTALLED__ = False
__IS_SUPERVISION_INSTALLED__ = False

try:
    import supervision
    import ultralytics

    __IS_ULTRALYTICS_INSTALLED__ = True
    __IS_SUPERVISION_INSTALLED__ = True
except ImportError:
    pass

from ..utils import Registry


PIPELINES = Registry(name='pipeline')


def build_pipeline(config: dict):
    pipeline = PIPELINES.build(config=config)
    return PIPELINES['Compose'](pipeline)

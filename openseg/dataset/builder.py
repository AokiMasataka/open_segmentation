from openback.utils import Registry


PIPELINES = Registry(name='pipeline')
DATASETS = Registry(name='Dataset')


def build_pipeline(config: dict):
    pipeline = PIPELINES.build(config=config)
    return PIPELINES['Compose'](pipeline)


def build_dataset(config: dict):
    return DATASETS.build(config=config)

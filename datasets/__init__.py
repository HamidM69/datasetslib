dataset_root = '/Users/armando/datasets'

def load_dataset(name, size='small', test_with_fake_data=False):
    """Loads dataset by name.
    Args:
      name: Name of the dataset to load.
      size: Size of the dataset to load.
      test_with_fake_data: If true, load with fake dataset.
    Returns:
      Features and labels for given dataset. Can be numpy or iterator.
    Raises:
      ValueError: if `name` is not found.
    """
    """
    if name not in DATASETS:
        raise ValueError('Name of dataset is not found: %s' % name)
    if name == 'dbpedia':
        return DATASETS[name](size, test_with_fake_data)
    else:
        return DATASETS[name]()
    """

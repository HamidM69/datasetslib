import sys, os
from six.moves.urllib.request import urlretrieve

def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent

def maybe_download(filename, dirname, source_url, force = False):
    """Download the data from source url, unless it's already here.
    Args:
        filename: string, name of the file in the directory.
        dirname: string, path to working directory.
        source_url: url to download from if file doesn't exist.
    Returns:
        Path to resulting file.
    """
    dest_filename = os.path.join(dirname, filename)
    if force or not os.path.exists(dest_filename):
        print('Downloading:', filename)
        filename, _ = urlretrieve(source_url + filename, dest_filename, reporthook=download_progress_hook)
        print('Download Complete!')
    statinfo = os.stat(dest_filename)
    print(filename,'(',statinfo.st_size,'bytes)')
    return dest_filename

def maybe_extract(filename, dirname, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(dirname)
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders
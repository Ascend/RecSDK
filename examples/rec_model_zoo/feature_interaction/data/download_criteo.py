import os
import logging
import zipfile
import urllib.request
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


if __name__ == "__main__":
    if not os.path.exists('./criteo/'):
        os.mkdir('./criteo/')
        logging.info("Begin to download criteo data, the total size is 3GB...")

    url = ''
    download(url, './criteo/criteo.zip')
    logging.info("Unzipping criteo dataset...")
    with zipfile.ZipFile('./criteo/criteo.zip', 'r') as zip_ref:
        zip_ref.extractall('./criteo/')
    logging.info("Done.")

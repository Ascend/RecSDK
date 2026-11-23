import os
import re
import logging
import zipfile
import urllib.request
from urllib.parse import urlparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DownloadProgressBar(tqdm):
    def __init__(self):
        self.total = None

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def is_valid_url(url):
    """基础URL校验函数"""
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):  # 必须包含协议和域名
            return False
        return re.match(
            r'^https?://([\w\-]+\.)+[\w\-]+(:\d+)?(/[\w\-./?%&=]*)?$', 
            url
        ) is not None
    except Exception as e:
        logger.error(f"check url fail: {str(e)}", exc_info=True)
        return False


def download(url, output_path):
    if not is_valid_url(url):
        logging.error(f"Invalid url: {url}")
        return
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

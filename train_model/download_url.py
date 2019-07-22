import os
import subprocess
import urllib.request
from tqdm import tqdm
from math import log


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def run_cmd(cmd):
    # start and process things, then wait
    p = subprocess.Popen(cmd.split())
    return p.communicate()


def download_and_unzip_from_url(url, output_dir):
    print(f'Downloading from {url}...')
    download_url(url, 'tmp.zip')
    print('Done! Unzipping...')
    run_cmd(f'unzip tmp.zip -d {output_dir}')
    print('Done!')
    run_cmd('rm tmp.zip')
    # Fix the corrupt EXIF headers
    run_cmd('jhead -mkexif {output_dir}/PetImages/Cat/*.jpg')
    run_cmd('jhead -mkexif {output_dir}/PetImages/Dog/*.jpg')
    run_cmd('jhead -de {output_dir}/PetImages/Cat/*.jpg')
    run_cmd('jhead -de {output_dir}/PetImages/Dog/*.jpg')


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

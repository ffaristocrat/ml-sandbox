import glob
import shutil
import os
from typing import List

import pandas as pd
from PIL import Image

from chan.utils import cluster


def process_image(image_file: str, size: int) -> List:
    with Image.open(image_file) as image:
        image = image.convert('RGB')
        image = image.resize((size, size))

        pixels = []
        for band in range(3):
            pixels.extend(list(image.getdata(band=band)))
        return pixels


def cluster_images(image_data: str, size: int, n_clusters: int=8):
    # column 1 is the file name
    columns = list(range(2, size * size * 3 + 2))
    df = pd.read_csv(image_data, index_col=0, header=None)
    df['cluster'] = cluster(df[columns], n_clusters=n_clusters)['cluster']
    return df


def main():
    image_id = 0
    image_data = 'image-data.csv'
    size = 75
    clusters = 8
    input_directory = '/Users/ffaristocrat/Desktop/screenshots/*.png'
    output_directory = './images'

    print('size', size)
    print('clusters', clusters)
    
    with open(image_data, 'wt') as f:
        for file in glob.glob(input_directory):
            image_id += 1
            pixels = process_image(file, size)
            string = f'{image_id},{file},' + ','.join(
                [str(p / 255.0) for p in pixels])
            print(string, file=f)

    df = cluster_images(image_data, size, n_clusters=clusters)
    df[[1, 'cluster']].to_csv('image-cluster.csv')

    df = pd.read_csv(
        'image-cluster.csv', names=['image_id', 'file', 'cluster'],
        skiprows=1)
    print(df['cluster'].value_counts(normalize=True))

    shutil.rmtree(output_directory, ignore_errors=True)
    for row in df.itertuples():
        cluster_file = f"{output_directory}/" \
                       f"c{row.cluster}/" \
                       f"{row.image_id}.png"
        os.makedirs(f"{output_directory}/c{row.cluster}/", exist_ok=True)

        shutil.copyfile(row.file, cluster_file)
    
    df['file'] = df['file'].str.replace(
        '/Users/ffaristocrat/Desktop/screenshots/', '')

    print(df[~df['file'].str.startswith('Screen')][['file', 'cluster']])


main()

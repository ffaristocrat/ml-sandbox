import glob

from PIL import Image

size = 25

input_directory = '/Users/ffaristocrat/Desktop/screenshots/*.png'
output_directory = './images/'


def process_image(image_file, image_id, f):
    with Image.open(image_file) as image:
        image = image.convert('RGB')
        image = image.resize((size, size))
        image.save(output_directory + f'{image_id}.png', format='PNG')
        
        pixels = []
        for band in range(3):
            pixels.extend(list(image.getdata(band=band)))
        string = f'{image_id},' + ','.join(
            [str(p / 255.0) for p in pixels])
        print(string, file=f)


def main():
    image_id = 0
    
    with open('image-data.csv', 'wt') as f:
        for file in glob.glob(input_directory):
            image_id += 1
            print(image_id, file)
            process_image(file, image_id, f)


main()

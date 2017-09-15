import sys
from PIL import Image

im = Image.open(sys.argv[1])
rgb_im = im.convert('RGB')
width, height = im.size
outputIm = Image.new('RGB',(width,height))
pixels = outputIm.load()

for i in range(width):
	for j in range(height):
		dividedByTwo = tuple(element//2 for element in rgb_im.getpixel((i,j)))
		pixels[i,j] = dividedByTwo

outputIm.save( "Q2.png" )
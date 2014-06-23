#!/usr/bin/python
# RGB Hitogram
# This script will create a histogram image based on the RGB content of
# an image. It uses PIL to do most of the donkey work but then we just
# draw a pretty graph out of it.
#
# May 2009,  Scott McDonough, www.scottmcdonough.co.uk
#

import os
import sys
import Image, ImageDraw, ImageStat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


imagepath = sys.argv[1]  # The image to build the histogram of


histHeight = 120            # Height of the histogram
histWidth = 256             # Width of the histogram
multiplerValue = 1        # The multiplier value basically increases
							# the histogram height so that love values
							# are easier to see, this in effect chops off
							# the top of the histogram.
showFstopLines = True       # True/False to hide outline
fStopLines = 5


# Colours to be used
backgroundColor = (51,51,51)    # Background color
lineColor = (102,102,102)       # Line color of fStop Markers 
red = (255,60,60)               # Color for the red lines
green = (51,204,51)             # Color for the green lines
blue = (0,102,255)              # Color for the blue lines

##################################################################################


img = Image.open(imagepath)

if len(sys.argv) > 2 and sys.argv[2] == "brightness_hist":
	backgroundColor = (255,255,255)    # Background color
	red = (51,51,51)    # Background color
	green = (51,51,51)    # Background color
	blue = (51,51,51)    # Background color
	width, height = img.size
	pix = img.load()
	for x in range(width):
		for y in range(height):
			brightness = max(pix[x, y])
			pix[x, y] = (brightness, brightness, brightness)

hist = img.histogram()
histMax = max(hist)                                     # comon color
#average = sum(hist)/len(hist)
#histMax = 0
#for i in hist:
#	if int(i) > 3*average: pass
#	elif int(i) > histMax:
#		histMax = int(i)

#histHeight = histMax
xScale = float(histWidth)/len(hist)                     # xScaling
yScale = float((histHeight)*multiplerValue)/histMax     # yScaling 

im = Image.new("RGBA", (histWidth, histHeight), backgroundColor)   
draw = ImageDraw.Draw(im)


# Draw Outline is required
if showFstopLines:    
	xmarker = histWidth/fStopLines
	x =0
	for i in range(1,fStopLines+1):
		draw.line((x, 0, x, histHeight), fill=lineColor)
		x+=xmarker
	draw.line((histWidth-1, 0, histWidth-1, 200), fill=lineColor)
	draw.line((0, 0, 0, histHeight), fill=lineColor)


# Draw the RGB histogram lines
x=0; c=0;
for i in hist:
	if int(i)==0: pass
	else:
		color = red
		if c>255: color = green
		if c>511: color = blue
		draw.line((x, histHeight, x, histHeight-i*yScale), fill=color)        
	if x>255: x=0
	else: x+=1
	c+=1


#resize the image
#resize_factor = 1
#im = im.resize((histWidth*resize_factor, histHeight*resize_factor), Image.NEAREST)



x,y = np.random.rand(2,10)

fig = plt.figure()
ax = fig.add_subplot(111)

im = ax.imshow(im,extent=[0,histWidth, 0,histHeight])

# Now save and show the histogram
output_file_name = os.path.splitext(imagepath)[0] + "_histogram.png"
plt.savefig(output_file_name)

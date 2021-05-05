blocks = int(input("Enter the number of blocks: "))
height = 0
inlayer = 1
while inlayer <= blocks:
	height += 1
	blocks -= inlayer
	inlayer += 1
100
print("Height of the pyramid:",height)
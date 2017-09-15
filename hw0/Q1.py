import sys

fr = open(sys.argv[1],'r')
reads = fr.read()
s = reads.split()
fr.close()

stringArr = []
indexArr = []
freqArr = []
index = 0

for element in s:
	if len(stringArr) != 0:
		for i in range(len(stringArr)):
			if i == (len(stringArr) - 1):
				if stringArr[i] != element:
					stringArr.append(element)
					indexArr.append(index)
					freqArr.append(1)
					index += 1
				else:
					freqArr[i] += 1
			else: 
				if stringArr[i] == element:
					freqArr[i] += 1
					break
	else:
		stringArr.append(element)
		indexArr.append(index)
		freqArr.append(1)
		index += 1

fw = open('Q1.txt','w')
for i in range(index):
	output = stringArr[i] +" "+ str(indexArr[i])+" "+ str(freqArr[i])
	fw.write(output)
	if i != (index - 1):
		fw.write('\n')
fw.close()

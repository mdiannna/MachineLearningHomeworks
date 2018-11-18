import numpy as np
from io import StringIO   # StringIO behaves like a file object


######################################
###  CONSTANTS 
######################################
FILE_NAME = "Marusic_I_Diana_train.csv"
ALL_FEATURES = ["Breed Name","Weight(g)","Height(cm)","Longevity(yrs)",
	"Energy level", "Attention Needs", "Coat Lenght", "Sex",
	"Owner Name"]

COLUMN_NAMES = ["Breed Name","Weight(g)","Height(cm)","Longevity(yrs)",
	"Energy level", "Attention Needs", "Coat Lenght", "Sex",
	"Owner Name"]

SELECTED_FEATURES_BREED = ["Weight(g)","Height(cm)",
	"Energy level", "Attention Needs", "Coat Lenght", "Sex",];

SEX = []
BREED_NAMES = []

def initValues():
	pass

def toNumericalData(line):
	# female, male = 0, 1
	# short, med, high = 0,1,2
	line = line.replace("female", "0")
	line = line.replace("male", "1")
	line = line.replace("short", "0")
	line = line.replace("med", "1")
	line = line.replace("long", "2")
	line = line.replace("low", "0")
	line = line.replace("high", "1")
	return line
	
def excludeCols(line):
	pass

def readData():
	f = open(FILE_NAME)
	line = f.readline()  # skip the header
	print(line)
	# data = np.loadtxt(f)

	for line in f:
		print(line)
		line = toNumericalData(line)
		print(line)
		text = np.genfromtxt(StringIO(line),dtype='str', delimiter=',')
		print(text)
		filtered_text = text[1:text.size-1]


		# TODO: replace missing values 
		
		print(filtered_text)
		print(np.array(filtered_text).astype(np.float))

	# np.loadtxt(StringIO(line), delimiter=",")
	# # print(data)
	# print(f.read())


def main():
	readData()

if __name__ == "__main__":
	main()

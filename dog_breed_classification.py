import numpy as np
from io import StringIO   # StringIO behaves like a file object
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

# SEX = []
# BREED_NAMES = []
breed_tags = {}

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
	# missing values => nan
	line = line.replace(",,", ',nan,')
	return line

def tagBreeds(breeds):

	breeds_set = set(breeds)
	print(breeds_set)
	global breed_tags;

	breed_tags = {};

	for idx, breed in enumerate(breeds_set):
		breed_tags[str(breed)] = idx
		
	tagged_breeds = []

	for breed in breeds:
		# print(breed)
		# print(breed_tags.get(breed))
		tagged_breeds.append(breed_tags.get(breed))

	return tagged_breeds


def readData():
	data = []
	breeds = []

	f = open(FILE_NAME)

	line = f.readline()  # skip the header
	print(line)
	# data = np.loadtxt(f)

	for line in f:
		# print(line)
		line = toNumericalData(line)
		# print(line)
		text = np.genfromtxt(StringIO(line),dtype='str', delimiter=',')
		# print(text)
		filtered_text = text[1:text.size-1]
		breeds.append(text[0])


		
		# print(filtered_text)
		dataline = np.array(filtered_text).astype(np.float)
		# print(dataline)
		data.append(dataline)

	labels = tagBreeds(breeds)


	return data, labels

	# np.loadtxt(StringIO(line), delimiter=",")
	# # print(data)
	# print(f.read())


# TODO: handle missing values 
# https://scikit-learn.org/stable/modules/impute.html
def handleMissing(data):
	# Available strategies: mean, median, most_frequent, constant
	imp = SimpleImputer(missing_values=np.nan, strategy='mean')
	imp.fit(data)  
	result = imp.transform(data)
	
	return result


def predictBreed(data, labels):
	# print(data[:3])
	# # data, nr coloanei(de ex. coloana 0), axis=1=>coloana
	# data1 = np.delete(data[:3],  0, 1)
	# print(data1)

	# print(data[:3])
	# print("---------")
	# Delete longevity column
	data_some_cols = np.delete(data,  2, 1)
	print(data_some_cols)

	# TODO: impart in training, validation, test
	# TODO: Linear Regression (Ridge, Lasso)
	# 		Logistic Regression
	#       Random Forests
	#       KNN
	#  find best hyper-parameters
	X = data_some_cols[0:100]
	y = labels[0:100]

	reg = LinearRegression().fit(X, y)
	reg.score(X, y)
	reg.coef_
	# reg.intercept

	# TODO: impart in training, validation, test
	X_test = data_some_cols[100:200]
	y_test = labels[100:200]

	y_pred = reg.predict(X_test).astype(int)
	print(y_pred)
	print(np.array(y_test))

	# TODO: choose what scores to use and what not
	# accuracy = accuracy_score(y_test, y_pred, normalize=False)
	accuracy = accuracy_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred, average="weighted")
	# f1 = f1_score(y_test, y_pred, average="macro")
	# f1 = f1_score(y_test, y_pred, average="micro")
	# f1 = f1_score(y_test, y_pred, average=None)
	precision = precision_score(y_test, y_pred, average="weighted")
	recall = recall_score(y_test, y_pred, average="weighted")
	print("-----accuracy score:-----")
	print(accuracy)

	print("-----f1 score:-----")
	print(f1)

	print("-----precision score:-----")
	print(precision)

	print("-----recall score:-----")
	print(recall)

def main():
	data, labels = readData()
	# print(data)
	# check nam
	# print(data[6])
	data = handleMissing(data)
	# check nan
	# print(data[6])

	# print(labels);
	predictBreed(data, labels);



if __name__ == "__main__":
	main()

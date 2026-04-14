import numpy as np
data = [
    [150, 7.0, 1, 'Apple'],
    [120, 6.5, 0, 'Banana'],
    [180, 7.5, 2, 'Orange'],
    [155, 7.2, 1, 'Apple'],
    [110, 6.0, 0, 'Banana'],
    [190, 7.8, 2, 'Orange'],
    [145, 7.1, 1, 'Apple'],
    [115, 6.3, 0, 'Banana']
]
features = []

for row in data:
  features.append(row[:3])

X=np.array(features)

indices = {"Apple":0,"Banana":1, "Orange":2}
rev_indices = {0:"Apple",1:"Banana", 2:"Orange"}

#one hot encoding the labels
one_hot_enc = []

for row in data:
  one_hot_enc.append(indices[row[3]])

y = np.array(one_hot_enc)

def euc_dist(a,b):
  return float(np.sqrt(np.sum((a-b)**2)))

class KNN:
  def __init__(self, k=3):
    self.k = k

  def fit(self,X,y):
    self.X = X
    self.y = y

  def predict_one(self, x):
    distance = []
    for i in range(self.X.shape[0]):
      distance.append((euc_dist(x, self.X[i]),self.y[i]))
    k_nearest = []
    distance.sort(key=lambda x: x[0])
    for i in range(self.k):
      k_nearest.append(distance[i])
    labels = []
    for i in range(self.k):
      labels.append(k_nearest[i][1])
    dictionary = {labels.count(0):0, labels.count(1):1, labels.count(2):2}
    max_freq = max(labels.count(0), labels.count(1), labels.count(2))
    prediction = dictionary[max_freq]
    return prediction
    
  def predict(self, X_test):
    predictions = []
    for i in range(len(X_test)):
      predictions.append(self.predict_one(X_test[i]))
    return predictions
  
test_data = np.array([
    [118, 6.2, 0],  # Expected: Banana
    [160, 7.3, 1],  # Expected: Apple
    [185, 7.7, 2]   # Expected: Orange
])

pred_dict = {0:"Apple", 1:"Banana", 2:"Orange"}
model = KNN(k=1)
model.fit(X,y)
list_1 = []
for i in range(len(test_data)):
    pred = model.predict_one(test_data[i])
    list_1.append(pred_dict[pred])
print(list_1)

model = KNN(k=3)
model.fit(X,y)
list_3 = []
for i in range(len(test_data)):
    pred = model.predict_one(test_data[i])
    list_3.append(pred_dict[pred])
print(list_3)

model = KNN(k=5)
model.fit(X,y)
list_5 = []
for i in range(len(test_data)):
    pred = model.predict_one(test_data[i])
    list_5.append(pred_dict[pred])
print(list_5)
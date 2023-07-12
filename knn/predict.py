import pickle
import numpy as np

classfier = pickle.load(open("KNN_model_10.gbsw", "rb"))

print(int(classfier.predict(np.array([[0.9645425136040272, 0.16826402817331804], [0.031862858743465915, 0.979687434615768]])).tolist()[1]))


def knn_predict(f1, f2):
  result = int(classfier.predict(np.array([[0.9645425136040272, 0.16826402817331804], [0.031862858743465915, 0.979687434615768]])).tolist()[1])
  if result == 3 : 
    return "fall"
  elif result == 2:
    return "sit"
  else :
    return "stand"
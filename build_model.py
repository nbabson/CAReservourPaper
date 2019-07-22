import sys
import csv
import numpy
from sklearn import linear_model

print('Building scikit model.')    
datafile = sys.argv[1] 
resultsfile = sys.argv[2]
data = open(datafile, 'rt') 
results = open(resultsfile, 'rt') 
reader = csv.reader(data, delimiter = ' ')
reader2 = csv.reader(results, delimiter = ' ')
x = list(reader)
d = list(x)
r = list(reader2)[0]    
res = [float(i) for i in r]
#r = r.astype(np.float64)
ans = []
for item in d: 
    ans.append(list(map(int, item)))


clf2 = linear_model.LinearRegression()    
clf2 = clf2.fit(ans, res)
predicted = clf2.predict(ans);    
    
with open(sys.argv[3], 'w') as f:
    for item in predicted:
        f.write("%.2f " % item)
f.close()	
data.close()
results.close()    

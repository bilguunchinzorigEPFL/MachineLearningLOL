import csv
def make_submission(name,y_pred):
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(range(0,len(y_pred))+1, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def data_loader(path,label):
    data=[]
    with open(path,'r') as target:
        rows=target.readlines()
        data=zip(rows,[label]*len(rows))
    return data

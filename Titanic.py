import pandas as pd
df = pd.read_csv("F:/Khai phá dữ liệu/Khai phá dữ liệu/titanic.csv")
print(df.head()) # xuat du lieu file csv
print(df.shape) # kich thuoc file csv
print("========================") 
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
 # khong lay(loai bo) du lieu cac cot tren
print(df.head()) # xuat du lieu dong 6
inputs = df.drop('Survived',axis='columns') # input khong lay(loai bo) du lieu cot survived
target = df[["Survived"]] # output chi lay du lieu cot survived
print("========================") 
#chuyển đổi cột dữ liệu Sex thành số
dummies = pd.get_dummies(inputs.Sex)
print(dummies.head(3)) # xuat du lieu dong 12
print("========================") 
#nối vào dữ liệu đầu
inputs = pd.concat([inputs,dummies],axis='columns') # noi du lieu inpus voi dummies 
print(inputs.head(3)) # xuat du lieu dong 16
print("========================")
inputs.drop(['Sex','male'],axis='columns',inplace=True) # inputs khong lay(loai bo) cot sex,male
print(inputs.head(3)) # xuat du lieu 19
print("========================")
inputs.columns[inputs.isna().any()]
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
print(inputs.head())
#chia tỉ lệ train-test
from sklearn.model_selection import train_test_split # thư viện chứa hàm chia dữ liệu 
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=35) # 80% train,20% test
print("Tập huấn luyện: ")
print(X_train.shape)
print(y_train.shape)
print("Tập test: ")
print(X_test.shape)
print(y_test.shape)
print("========================")
#------------------- thuật toán Naive Bayes  --------------
from sklearn.naive_bayes import GaussianNB
model1 = GaussianNB().fit(X_train,y_train)
y_pred1 = model1.predict(X_test)
# Đánh giá mô hình dựa trên kết quả dự đoán (với độ đo đơn giản Accuracy, Precision, Recall)
# In ra kết quả độ chính xác
from sklearn.metrics import accuracy_score
print("Accuracy Score:", accuracy_score(y_test, y_pred1))
print("Số lớp", model1.classes_)
#In ra kết quả độ chính xác trên từng lớp yes/no
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred1))
#In ra ma trận kết quả dự đoán
from sklearn.metrics import confusion_matrix
confusion_matrix1 = confusion_matrix(y_test,y_pred1)
print(confusion_matrix1)

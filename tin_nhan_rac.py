from google.colab import drive
import pandas as pd
import nltk
import string
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Mount Google Drive
drive.mount('/content/drive')

# Đọc dữ liệu từ file
file_url = '/content/drive/MyDrive/Colab Notebooks/SMSSpamCollection.txt'
data = pd.read_csv(file_url, sep='\t', header=None, names=["label", "sms"])
print(data.head())  # In ra 5 dòng đầu tiên của dữ liệu

# Tải tài nguyên NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Tạo danh sách stopwords và punctuation
stopwords = nltk.corpus.stopwords.words('english')
punctuation = string.punctuation

# Xử lý giá trị thiếu và đảm bảo dữ liệu dạng chuỗi
data = data.dropna(subset=['sms'])  # Loại bỏ các dòng có giá trị NaN
data['sms'] = data['sms'].astype(str)  # Đảm bảo tất cả dữ liệu trong cột 'sms' là chuỗi

# Hàm tiền xử lý đã sửa đổi
def pre_process(sms):
    # Loại bỏ dấu câu
    remove_punct = "".join([word.lower() for word in sms if word not in punctuation])

    # Token hóa
    tokenize = nltk.tokenize.word_tokenize(remove_punct)

    # In các từ đã token hóa để kiểm tra
    print("Tokenized:", tokenize)  # In ra các từ sau khi token hóa

    # Loại bỏ stopwords
    remove_stopwords = [word for word in tokenize if word not in stopwords]

    # In các từ sau khi loại bỏ stopwords để kiểm tra
    print("After removing stopwords:", remove_stopwords)

    # Kiểm tra nếu tin nhắn bị rỗng sau khi loại bỏ stopwords
    if len(remove_stopwords) == 0:
        return None  # Trả về None nếu không còn từ khóa nào sau tiền xử lý
    return " ".join(remove_stopwords)

# Tiền xử lý toàn bộ dữ liệu
data['processed'] = data['sms'].apply(pre_process)

# Kiểm tra số lượng tin nhắn còn lại sau tiền xử lý
print(f"Sau khi tiền xử lý, có {data['processed'].notna().sum()} tin nhắn còn lại.")
print("Một số tin nhắn sau tiền xử lý:")
print(data['processed'].head(10))  # In ra 10 dòng đầu tiên

# Kiểm tra lại dữ liệu sau khi loại bỏ tin nhắn quá ngắn
data = data.dropna(subset=['processed'])  # Loại bỏ tin nhắn rỗng sau tiền xử lý
data = data[data['processed'].str.len() > 3]  # Giảm độ dài tối thiểu của tin nhắn (>= 3 ký tự)

# Kiểm tra số lượng tin nhắn còn lại
print(f"Sau khi loại bỏ tin nhắn quá ngắn, có {data['processed'].notna().sum()} tin nhắn còn lại.")

# Chuyển tin nhắn thành vector sử dụng TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['processed'])  # Chuyển các tin nhắn thành các vector TF-IDF
y = data['label']  # Nhãn (spam/ham)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình KNN và huấn luyện
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Đánh giá mô hình trên tập kiểm tra
y_pred = knn.predict(X_test)
print("Kết quả phân loại:\n")
print(classification_report(y_test, y_pred))  # In kết quả phân loại

# Hàm dự đoán cho tin nhắn mới
def predict_knn(sms):
    sms_processed = pre_process(sms)  # Tiền xử lý tin nhắn đầu vào
    if sms_processed is None:
        return "Tin nhắn không hợp lệ"  # Nếu tin nhắn trống sau tiền xử lý
    sms_vec = vectorizer.transform([sms_processed])  # Chuyển tin nhắn thành vector
    prediction = knn.predict(sms_vec)  # Dự đoán nhãn của tin nhắn
    return prediction[0]  # Trả về nhãn của tin nhắn (spam/ham)

# Kiểm tra với tin nhắn đầu vào
user_input = input("Vui lòng gõ tin nhắn spam hoặc ham để kiểm tra xem chức năng của chúng tôi có dự đoán chính xác không: \n")
result = predict_knn(user_input)  # Dự đoán tin nhắn
print(f'Kết quả phân loại: {result}')
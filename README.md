<div align="center">

# 🕵️ Fake News Detection  
### **Detect Fake News using DistilBERT + Explainable AI (Integrated Gradients)**  

![Shield](https://img.shields.io/badge/NLP-Fake%20News%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Model](https://img.shields.io/badge/Model-DistilBERT-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

</div>

## 📌 **Mục lục**
- [1. Giới thiệu](#-giới-thiệu)
- [2. Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [3. Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [4. Tính năng chính](#-tính-năng-chính)
- [5. Cài đặt & Chạy ứng dụng](#-cài-đặt--chạy-ứng-dụng)
- [6. Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [7. Giải thích mô hình (XAI)](#-giải-thích-mô-hình-xai)
- [8. Demo giao diện](#-demo-giao-diện)
- [9. Tài liệu liên quan](#-tài-liệu-liên-quan)
- [10. Tác giả](#-tác-giả)

---

## 🔍 **Giới thiệu**

Fake news đang là một vấn đề nghiêm trọng trên Internet.  
Dự án này xây dựng hệ thống **phát hiện tin giả** dựa trên:

- **DistilBERT** (một phiên bản nhẹ của BERT, nhanh và hiệu quả)
- Phương pháp giải thích mô hình **Integrated Gradients (Captum)**  
- Giao diện người dùng trực quan bằng **Streamlit**

Ứng dụng hiển thị:
- Xác suất Real News / Fake News  
- Kết luận cuối cùng  
- Tô màu những từ ảnh hưởng mạnh đến quyết định của mô hình  

Giúp người dùng **hiểu vì sao AI đưa ra dự đoán** — rất quan trọng trong lĩnh vực XAI.

---

## 🧰 **Công nghệ sử dụng**

| Công nghệ | Mô tả |
|----------|-------|
| **DistilBERT** | Model NLP pretrained từ HuggingFace |
| **PyTorch** | Huấn luyện & inference model |
| **Captum** | Explainable AI – Integrated Gradients |
| **Streamlit** | Xây dựng giao diện Web App |
| **Transformers** | Xử lý tokenizer + inference |
| **Numpy** | Xử lý số liệu |

---

## 🧱 **Kiến trúc hệ thống**

User Input → Tokenizer → DistilBERT → Softmax Output
↓
Integrated Gradients (Captum)
↓
Highlight Words

yaml
Copy code

---

## ⭐ **Tính năng chính**

- ✔ Phát hiện tin giả *dựa trên văn bản tiếng Anh*
- ✔ Hiển thị xác suất:
  - 🟩 Real News  
  - 🟥 Fake News  
- ✔ Giải thích mô hình bằng **Integrated Gradients**
- ✔ Tô màu từ theo mức độ ảnh hưởng:
  - 🔴 tăng xác suất Fake News  
  - 🔵 tăng xác suất Real News  
- ✔ Giao diện đơn giản, dễ dùng

---

## ⚙ **Cài đặt & Chạy ứng dụng**

### 1️⃣ Clone repo
```bash
git clone https://github.com/Hoai-Nam31/fake-news-detection.git
cd fake-news-detection
2️⃣ Cài đặt thư viện
bash
Copy code
pip install -r requirements.txt
Tạo file requirements.txt nếu chưa có:

nginx
Copy code
streamlit
torch
transformers
captum
numpy
3️⃣ Chạy app
bash
Copy code
streamlit run app.py
Ứng dụng chạy tại:

arduino
Copy code
http://localhost:8501
📁 Cấu trúc thư mục
php
Copy code
fake-news-detection/
│── app.py                     # Web App (Streamlit)
│── distilBert_model.ipynb     # Notebook huấn luyện model
│── distilbert_final/          # Model fine-tuned (ignored)
│── README.md                  # Mô tả dự án
│── report-final.pptx          # File slide báo cáo
│── Fake news detection final.pdf   # File báo cáo PDF
⚠ Thư mục model không nên đẩy lên GitHub -> cần ignore.

🧠 Giải thích mô hình (XAI)
Dự án sử dụng Integrated Gradients để truy ngược mức độ ảnh hưởng của từng token.

Công thức:

ini
Copy code
IG = (input - baseline) × ∫ (d model(input_scaled) / d input_scaled)
Baseline dùng:

[PAD] hoặc [CLS] token

Hệ thống tô màu:

🔴 Từ đẩy mô hình về lớp Fake News

🔵 Từ kéo mô hình về lớp Real News

📑 Tài liệu liên quan
report-final.pptx – bản slide

Fake news detection final.pdf – bản báo cáo

Notebook: distilBert_model.ipynb

<div align="center">
⭐ Nếu bạn thấy dự án hữu ích, hãy để lại một ⭐ trên GitHub!
</div> ```
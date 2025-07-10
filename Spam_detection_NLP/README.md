 ![image](https://github.com/user-attachments/assets/27b60c40-72ce-42f2-8e07-4590842a78d3)


## ⚙️ Steps to Build the Spam Detection Model

1. 📥 **Data Collection**  
   Gather emails with a mix of `spam` and `ham` (non-spam) messages.

2. 🧠 **Feature Engineering**  
   Extract custom features using regex:
   - 🔗 Detect presence of links (URLs)  
   - 📞 Identify phone numbers

3. 🧹 **Text Preprocessing**  
   - Convert to lowercase  
   - Remove stop words and punctuation  
   - Strip noise like links and phone numbers

4. ✂️ **Tokenization**  
   Break down each email into individual tokens (words) for analysis.

5. 🔢 **Vectorization**  
   Convert tokens into numerical vectors using `CountVectorizer` – this is the input to the model.

6. 🤖 **Model Training & Evaluation**  
   - Train a **Multinomial Naive Bayes** classifier  
   - Learn to predict whether an email is `SPAM` or `NOT SPAM`  
   - Evaluate performance using accuracy and classification metrics

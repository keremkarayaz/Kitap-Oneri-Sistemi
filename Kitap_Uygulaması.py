import pandas as pd
import streamlit as st
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split


# Verileri yükleyin
world_classics_df = pd.read_csv('world_classics_books.csv')
books_by_category_df = pd.read_excel('all_books_by_category.csv')

# Streamlit başlığı ve açıklama
st.title("Kitap Öneri Sistemi")
st.write("Okuduğunuz dünya klasiklerine göre size öneriler sunalım!")

# Kullanıcının okuduğu dünya klasiklerini seçmesi için bir seçim kutusu
user_books = st.multiselect(
    "Lütfen okuduğunuz dünya klasiklerini seçin:",
    world_classics_df['Title'].tolist()
)

if user_books:
    # Kullanıcının seçtiği kitapların bilgilerini alalım
    user_books_info = world_classics_df[world_classics_df['Title'].isin(user_books)]

    # Kullanıcının okuduğu kitapların türlerini ve sayfa sayısının ortalamasını alıyoruz
    user_books_genres = user_books_info['Categories'].unique()
    user_books_page_count = user_books_info['Page Count'].mean()  # Ortalama sayfa sayısı

    # Kullanıcının okuduğu kitaplara puan veriyoruz (4.0 olarak varsayalım)
    user_data = []
    for book in user_books:
        user_data.append([1, book, 4.0])  # 1: Varsayılan kullanıcı ID'si, 4.0: Beğendiği puan

    # Veri çerçevesini oluşturma ve Surprise formatına dönüştürme
    user_ratings_df = pd.DataFrame(user_data, columns=['User', 'Title', 'Rating'])
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(user_ratings_df, reader)

    # Eğitim ve test seti oluşturma
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Modeli eğitme
    algo = KNNBasic()
    algo.fit(trainset)

    # Kullanıcının okumadığı kitaplardan öneri yapma
    predictions = []
    for book_title in books_by_category_df['Title'].unique():
        if book_title not in user_books:
            prediction = algo.predict(1, book_title)
            predictions.append((book_title, prediction.est))

    # Kullanıcının okuduğu kitapların türüne ve sayfa sayısına göre öneriler yapılacak
    recommended_books = books_by_category_df[books_by_category_df['Categories'].isin(user_books_genres)]
    recommended_books['PageDiff'] = abs(recommended_books['Page Count'] - user_books_page_count)

    # Sayfa sayısına göre sıralama
    recommended_books = recommended_books.sort_values(by='PageDiff').head(10)

    st.subheader("Önerilen Kitaplar:")
    for index, row in recommended_books.iterrows():
        st.write(f"{row['Title']} - Tür: {row['Categories']} - Sayfa Sayısı: {row['Page Count']}")

else:
    st.write("Lütfen en az bir kitap seçin!")



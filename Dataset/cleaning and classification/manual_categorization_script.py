import streamlit as st
import pandas as pd

# Load your CSV file
df = pd.read_csv('news_categorized_manually.csv')

# Predefined categories
categories_list = ["Operation Sindoor", "Ayodhya Ram Mandir", "Covaxin", "Chandrayaan-3", "Other"]

# initialized required session states
if 'index' not in st.session_state:
    st.session_state.index = 338
if 'categories' not in st.session_state:
    st.session_state.categories = df['category']
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = st.session_state.categories[st.session_state.index]

st.title("Manual News Categorization")

st.subheader(f"Article {st.session_state.index} of {len(df)}")

# Current article
article = df.iloc[st.session_state.index]
st.subheader(f"Headline: {article['headline']}")
st.write(f"News: {article['news']}")
st.write(f"{article['link']}")

#category selection
st.session_state.selected_category = st.selectbox(
    "Select Category",
    options=categories_list,
    index=categories_list.index(st.session_state.selected_category)
)

if st.button("Save & Next"):
    st.session_state.categories[st.session_state.index-1] = st.session_state.selected_category

    df['category'] = st.session_state.categories
    df.to_csv('news_categorized_manually.csv', index=False)
    st.success("Saved successfully as 'news_categorized_manually.csv'")

    if st.session_state.index < len(df) - 1:
        st.session_state.index += 1

df_preview = df.copy()
df_preview['category'] = st.session_state.categories
st.dataframe(df_preview['category'])
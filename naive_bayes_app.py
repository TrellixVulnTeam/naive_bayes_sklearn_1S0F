import streamlit as st

import homepage
import math_background
import naive_bayes_interactive_examples_page

pages = {
    'Home': homepage,
    'Mathematics Background': math_background,
    'Naive-Bayes with sklearn': naive_bayes_interactive_examples_page,
}

def main_app():

    st.set_page_config(page_title='Naive-Bayes Classification', page_icon=None, layout='wide', initial_sidebar_state='auto')
    
    # Sidebar navigation
    st.sidebar.title('Page Navigation')
    page_selection = st.sidebar.radio('', list(pages.keys()))

    page = pages[page_selection]
    page.app_page()

# Run main function
if __name__ == "__main__":
    main_app()
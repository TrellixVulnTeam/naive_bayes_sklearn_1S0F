import streamlit as st


def button_created(key):
    """
    Callback function for initializing buttons in streamlit
    """
    if key+'_dict' not in st.session_state:
        st.session_state[key+'_dict'] = {'was_created': True, 'was_pressed': False}

def button_changed(key):
    """
    Callback function for changing a button that controls type of output.
    Sets the 'input_changed' key to True for the corresponding Run Button
    """
    if key+'_dict' not in st.session_state:
        st.session_state[key+'_dict'] = {'was_created': True, 'was_pressed': False, 'input_changed': False}
    else:
        st.session_state[key+'_dict']['input_changed'] = True
import streamlit as st
from services import upload_data, get_response
import uuid

# Title
st.title("Conversational RAG System")

# File Upload Section
st.header("Upload CSV Data")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    if st.button("Upload File"):
        try:
            # Call the upload_data function from services.py
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            upload_data(file_path=uploaded_file.name)
            st.success("File uploaded and processed successfully!")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Chat Interface Section
st.header("Chat with the System")

# Generate a unique session ID for the user
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

st.sidebar.header("Session Management")
st.sidebar.write(f"Session ID: {st.session_state['session_id']}")

st.write("Ask the chatbot anything!")

user_input = st.text_input("Your Question:", key="user_input")

if st.button("Send") or user_input:
    if user_input:
        with st.spinner("Processing..."):
            try:
                # Call the get_response function from services.py
                response = get_response(session_id=st.session_state["session_id"], user_input=user_input)
                st.write(f"Bot: {response}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question!")

# Footer
st.markdown("---")
st.caption("Conversational RAG System powered by Streamlit and LangChain")

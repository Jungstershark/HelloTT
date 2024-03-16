import streamlit as st

#function that provides feedback
def feedback():
    positive_feedback = "Thank you! We're happy to help!"
    negative_feedback = "We're sorry to hear that! Please contact a representative if your query is not answered!"

    if 'feedback_submitted' not in st.session_state:
        st.session_state['feedback_submitted'] = False
    if st.session_state['feedback_submitted']:
        st.success("Feedback already submitted!")
    else:
        feedback = st.selectbox("Was the answer helpful?", ["Select", "Yes", "No"])
        #there is feedback!
        if feedback != "Select":
            comment = st.text_area("Comments (optional):")
            if st.button("Submit Feedback"):
                if feedback=="Yes":
                    st.write(positive_feedback)
                    st.session_state['feedback_submitted'] = True
                elif feedback=="No":
                    st.write(negative_feedback)
                    st.session_state['feedback_submitted'] = True
            return comment
        #no feedback
        return False

    
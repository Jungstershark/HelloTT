import smtplib
from email.message import EmailMessage
import streamlit as st

'''
IMPORTANT: TECHNICALLY NOT WORKING DUE TO Gmail change in security policy if not wrong
However, with the file_data and screenshots, can easily transmit them elsewhere/
'''


def send_email(issue, screenshot_files):
    sender_email = "helloTTpoc@gmail.com"
    sender_password = "@dmin123"  
    receiver_email = "shummymomo123@gmail.com"
    
    msg = EmailMessage()
    msg['Subject'] = 'New Issue Reported'
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content(issue)
    
    if screenshot_files:
        for screenshot_file in screenshot_files:
            file_data = screenshot_file.read()
            file_name = screenshot_file.name
            msg.add_attachment(file_data, maintype='image', subtype='png', filename=file_name)

    with smtplib.SMTP_SSL("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(sender_email, sender_password)  
        smtp.send_message(msg)

def contact(screenshots):
    if 'contact_submitted' not in st.session_state:
        st.session_state['contact_submitted'] = False
    if st.session_state['contact_submitted']:
        st.success("Your query has been sent to our team.")
    else:
        if st.button("Contact a Representative"):
            st.write("You can contact our team at <PLACEHOLDER EMAIL HERE>.")
            st.write("1. Upload any screenshots of the issue in the sidebar (Optional)")
            issue = st.text_area("2. Describe the issue you faced")
            if st.button("Send to Email"):
                st.session_state['contact_submitted'] = True
                send_email(issue, screenshots)
                return True
    return False


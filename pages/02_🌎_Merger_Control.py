"""
This page doesn't use genAI, but it was decided as very important in the business use case :0
-faster and more accurate to read and output data from an excel file using pandas than genai lol
"""

import pandas as pd
import streamlit as st
# import random
import time

st.title("🌎 Merger_Control")

# # Streamed response emulator
# def response_generator(text):
#     response = random.choice(
#         [
#             text
#         ]
#     )
#     for word in response.split():
#         yield word + " "
#         time.sleep(0.08)


with st.chat_message("assistant", avatar="🤖"):
    st.markdown("Welcome to  Hello TT - *Merger Control Edition*. I am here to support your 🗺️ ***anti-trust multi jurisdiction assessment***.")

    st.markdown("The following information pertains to Temasek Group merger control revenue for the financial year ended 31st March 2023.")


# with st.chat_message("assistant", avatar="🤖"):
#     st.write_stream(response_generator("Welcome to  Hello TT - *Merger Control Edition*. I am here to support your 🗺️ ***anti-trust multi jurisdiction assessment***."))

#     time.sleep(1)

#     st.write_stream(response_generator("The following information pertains to Temasek Group merger control revenue for the financial year ended 31st March 2023."))


# Upload CSV through the sidebar
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    # ASSUME FILE FORMATS SAME
    df = pd.read_csv(uploaded_file, header=5)

    #list of countries to dynamically generate in select box:
    countries = df.iloc[:, 0].dropna().tolist()

    #for user to create new groups
    with st.sidebar:
        group_name = st.text_input("Name your new jurisdiction:")
        selected_countries_for_group = st.multiselect("Select countries for jurisdiction:", countries)
        #saving the custom group
        if st.button("Save Group"):
            #note that you cannot have duplicate groups with the same name-overriden here
            if group_name and selected_countries_for_group:
                st.session_state[group_name] = selected_countries_for_group
                st.success(f"Group '{group_name}' saved successfully!")
            else:
                st.error("Please specify a group name and select at least one country.")

        group_keys = [key for key in st.session_state.keys() if key not in ['__streamlit__', 'key', 'running_script']]
        #for user to delete groups
        selected_group_to_delete = st.selectbox("Select a jurisdiction to delete:", ['Select a group'] + group_keys)
        if st.button("Delete Group"):
            if selected_group_to_delete != 'Select a group':
                del st.session_state[selected_group_to_delete]
                st.success(f"Group '{selected_group_to_delete}' deleted successfully!")
            else:
                st.error("Please select a group to delete.")

    # select country/grp
    group_keys = [key for key in st.session_state.keys() if key not in ['__streamlit__', 'key', 'running_script']]
    selected_country_or_group = st.selectbox("Select a jurisdiction:", ['Select a jurisdiction'] + countries + group_keys)

    input_value = st.number_input("Revenue threshold to be verified (S$million):", min_value=0, step=1, format='%d')

    if st.button('Submit'):
        #singular country
        if selected_country_or_group in countries:
            country_value = str(df[df.iloc[:, 0] == selected_country_or_group].iloc[0, 1])
            country_value = country_value.replace(",","")
            
            if pd.isnull(country_value):
                st.markdown("There is no data for the jurisdiction you've selected!")
            else:
                # convert the country's value to hundred millions
                country_value_in_hundred_millions = int(country_value) 
                updated_input_value = input_value 
                if updated_input_value > country_value_in_hundred_millions:
                    st.markdown(f"**Does not exceed** threshold.")
                else:
                    st.markdown(f"**Exceeds** threshold.")

        #many countries(group)
        elif selected_country_or_group in group_keys:
            country_value = 0
            invalid_data_countries = []

            for country in st.session_state[selected_country_or_group]:
                country_value_str = str(df[df.iloc[:, 0] == country].iloc[0, 1])
                country_value_str = country_value_str.replace(",", "").strip()

                try:
                    country_value += int(country_value_str)
                except ValueError:
                    invalid_data_countries.append(country)
                    continue
            
            #if invalid data, one of country in group has no data.
            #display all invalid countries
            if invalid_data_countries:
                st.warning(f"Invalid data for: {', '.join(invalid_data_countries)}.")
            else:
                if country_value > 0:
                    updated_input_value = input_value

                    if updated_input_value > country_value:
                        st.markdown(f"**Does not exceed** threshold.")
                    else:
                        st.markdown(f"**Exceeds** threshold.")
                else:
                    st.error("Unable to perform comparison due to missing or invalid data.")

        else:
            st.warning("Please select a jurisdiction.")
else:
    st.sidebar.write("Please upload a CSV file.")


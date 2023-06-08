import streamlit as st
import requests
import time
import pandas as pd

# API endpoint URL
API_URL = 'http://localhost:5000/loan_prediction'


def main() :
    st.title('Dashboard demande de prêt')

    id_widget = st.sidebar.empty()
    client_id = id_widget.text_input("Enter your client ID")

    if client_id != '':
        id_widget.empty()
        st.sidebar.write(client_id)

    if client_id:
        # Make a request to the API
        response = requests.post(API_URL, json={'loan_id': client_id})

        if response.status_code == 200:
            if len(response.text)==0:
                return "La requête a échoué, veuillez vérifier votre saisie."

            prediction = response.json()

            # Display loan prediction
            st.subheader('Loan Prediction')
            st.write(prediction)

            ## Display client data
            #st.subheader('Client Data')
            #client_data = pd.DataFrame(data['CODE_GENDER'])
            #st.write(client_data)
        else:
            st.error('Error retrieving loan prediction. Please try again.')
            st.text(response.status_code)


if __name__ == '__main__':
    main()

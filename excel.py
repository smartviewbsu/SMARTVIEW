import os
import time
from openpyxl import Workbook
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID = "1nnY6VrGRVhLlsm9XrcfwP79IltYj7JzynMJ1gbXSgWU"
credentials = None

# Check if token.json exists to retrieve credentials
if os.path.exists("token.json"):
    credentials = Credentials.from_authorized_user_file("token.json", SCOPES)

# If credentials do not exist or are not valid, perform authentication
if not credentials or not credentials.valid:
    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        credentials = flow.run_local_server(port=0)

    # Save updated credentials to token.json
    with open("token.json", "w") as token:
        token.write(credentials.to_json())


def google_sheet_raw_data(cell1, cell2, cell3, compliant, non_compliant):
    """
    Update Google Sheets with raw data.

    Args:
        cell1 (str): The cell address for current date.
        cell2 (str): The cell address for compliant data.
        cell3 (str): The cell address for non-compliant data.
        compliant (str): Compliant data to be updated.
        non_compliant (str): Non-compliant data to be updated.
    """
    try:
        service = build("sheets", "v4", credentials=credentials)
        sheets = service.spreadsheets()

        current_date = datetime.now().strftime("%B %d, %Y")
        data1 = [[current_date]]
        data2 = [[compliant]]
        data3 = [[non_compliant]]

        body1 = {"values": data1}
        body2 = {"values": data2}
        body3 = {"values": data3}

        # Update Google Sheets with raw data
        result1 = sheets.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=cell1,
            body=body1,
            valueInputOption="RAW"
        ).execute()

        result2 = sheets.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=cell2,
            body=body2,
            valueInputOption="RAW"
        ).execute()

        result3 = sheets.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=cell3,
            body=body3,
            valueInputOption="RAW"
        ).execute()

        print(f"{result1.get('updatedCells')} cells updated with the current date: {data1}.")
        print(f"{result2.get('updatedCells')} cells updated with the value {data2}.")
        print(f"{result3.get('updatedCells')} cells updated with the value {data3}.")

    except HttpError as error:
        print(f"An error occurred: {error}")


def google_sheet_total_data():
    """
    Update Google Sheets with total data.
    """
    try:
        service = build("sheets", "v4", credentials=credentials)
        sheets = service.spreadsheets()

        # Retrieve data from specified ranges
        result = sheets.values().batchGet(spreadsheetId=SPREADSHEET_ID, ranges=['C4:C16', 'D4:D16']).execute()
        values = result.get('valueRanges', [])

        if len(values) == 2 and 'values' in values[0] and 'values' in values[1]:
            data1_sum = sum(float(val[0]) for val in values[0]['values'] if val)
            data2_sum = sum(float(val[0]) for val in values[1]['values'] if val)
        else:
            print("Unable to retrieve values from specified ranges.")
            return

        range1 = 'C17'
        range2 = 'D17'
        data1 = [[data1_sum]]
        data2 = [[data2_sum]]

        body1 = {"values": data1}
        body2 = {"values": data2}

        # Update Google Sheets with total data
        result1 = sheets.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=range1,
            body=body1,
            valueInputOption="RAW"
        ).execute()

        result2 = sheets.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=range2,
            body=body2,
            valueInputOption="RAW"
        ).execute()

        print(f"{result1.get('updatedCells')} cells updated with the sum of values from C4:C16: {data1_sum}.")
        print(f"{result2.get('updatedCells')} cells updated with the sum of values from D4:D16: {data2_sum}.")

    except HttpError as error:
        print(f"An error occurred: {error}")


def download_google_sheet_and_save():
    """
    Download Google Sheets and save as Excel file.
    """
    try:
        service = build("sheets", "v4", credentials=credentials)
        sheets = service.spreadsheets()

        # Retrieve data from Google Sheets
        result = sheets.values().get(spreadsheetId=SPREADSHEET_ID, range="A1:Z").execute()
        values = result.get('values', [])

        if not values:
            print('No data found.')
            return

        # Create Excel workbook and worksheet
        wb = Workbook()
        ws = wb.active

        # Populate Excel worksheet with retrieved data
        for row in values:
            ws.append(row)

        current_date = datetime.now().strftime("%Y-%m-%d")
        file_name = f"SMARTVIEW-{current_date}.xlsx"

        # Save Excel workbook
        wb.save(file_name)
        print(f'Sheet downloaded and saved successfully as "{file_name}".')

    except HttpError as error:
        print(f"An error occurred: {error}")


def clear_google_sheet_data():
    """
    Clear data from specified ranges in Google Sheets.
    """
    try:
        service = build("sheets", "v4", credentials=credentials)
        sheets = service.spreadsheets()

        ranges_to_clear = [
            'C1',
            'C4:C17',
            'D4:D17'
        ]

        # Clear data from specified ranges
        for range_ in ranges_to_clear:
            sheets.values().clear(spreadsheetId=SPREADSHEET_ID, range=range_).execute()
            print(f"Data cleared from range: {range_}")

    except HttpError as error:
        print(f"An error occurred: {error}")

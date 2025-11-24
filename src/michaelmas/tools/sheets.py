import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain_core.tools import tool

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def get_sheets_service():
    """
    Authenticates with the Google Sheets API and returns a service object.
    Handles the OAuth2 flow, prompting the user for authorization if necessary.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # You must have a credentials.json file from Google Cloud Console
            if not os.path.exists("credentials.json"):
                raise FileNotFoundError(
                    "credentials.json not found. Please follow the instructions in README.md to set it up."
                )
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    
    try:
        service = build("sheets", "v4", credentials=creds)
        return service
    except HttpError as err:
        print(err)
        return None

@tool
def create_spreadsheet(title: str) -> str:
    """
    Creates a new Google Sheet with the given title.
    Returns the ID and URL of the new spreadsheet.
    """
    service = get_sheets_service()
    if not service:
        return "Failed to get Google Sheets service."
    
    spreadsheet = {"properties": {"title": title}}
    try:
        spreadsheet = (
            service.spreadsheets()
            .create(body=spreadsheet, fields="spreadsheetId,spreadsheetUrl")
            .execute()
        )
        return f"Spreadsheet created with ID {spreadsheet.get('spreadsheetId')} and URL {spreadsheet.get('spreadsheetUrl')}"
    except HttpError as err:
        return f"An error occurred: {err}"

@tool
def get_sheet_data(spreadsheet_id: str, range_name: str) -> list | str:
    """
    Gets data from a specific range in a Google Sheet.
    `range_name` should be in A1 notation (e.g., 'Sheet1!A1:B5').
    """
    service = get_sheets_service()
    if not service:
        return "Failed to get Google Sheets service."
    try:
        sheet = service.spreadsheets()
        result = (
            sheet.values()
            .get(spreadsheetId=spreadsheet_id, range=range_name)
            .execute()
        )
        values = result.get("values", [])
        return values
    except HttpError as err:
        return f"An error occurred: {err}"

@tool
def update_sheet_data(spreadsheet_id: str, range_name: str, values: list[list]) -> str:
    """
    Updates a specific range in a Google Sheet with new data.
    `range_name` should be in A1 notation (e.g., 'Sheet1!A1:B5').
    `values` should be a list of lists, where each inner list is a row.
    """
    service = get_sheets_service()
    if not service:
        return "Failed to get Google Sheets service."
    try:
        body = {"values": values}
        result = (
            service.spreadsheets()
            .values()
            .update(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption="USER_ENTERED",
                body=body,
            )
            .execute()
        )
        return f"{result.get('updatedCells')} cells updated."
    except HttpError as err:
        return f"An error occurred: {err}"

@tool
def append_to_sheet(spreadsheet_id: str, range_name: str, values: list[list]) -> str:
    """
    Appends rows of data to a sheet. `range_name` is typically just the sheet name (e.g., 'Sheet1').
    `values` should be a list of lists, where each inner list is a row to append.
    """
    service = get_sheets_service()
    if not service:
        return "Failed to get Google Sheets service."
    try:
        body = {"values": values}
        result = (
            service.spreadsheets()
            .values()
            .append(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption="USER_ENTERED",
                body=body,
            )
            .execute()
        )
        return f"Appended data to {result.get('updates').get('updatedRange')}"
    except HttpError as err:
        return f"An error occurred: {err}"

# You will need to create this file based on the instructions in the README.
# This is just a placeholder to prevent errors if the file doesn't exist yet.
if not os.path.exists("credentials.json"):
    with open("credentials.json", "w") as f:
        f.write(
"""
{
    "installed": {
        "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
        "project_id": "YOUR_PROJECT_ID",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": "YOUR_CLIENT_SECRET",
        "redirect_uris": ["http://localhost"]
    }
}
"""
        )

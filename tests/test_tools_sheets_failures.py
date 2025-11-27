import pytest
from unittest.mock import patch
from michaelmas.tools.sheets import create_spreadsheet, get_sheet_data, update_sheet_data, append_to_sheet

def test_sheets_service_failure():
    with patch("michaelmas.tools.sheets.get_sheets_service", return_value=None):
        assert create_spreadsheet.invoke("title") == "Failed to get Google Sheets service."
        assert get_sheet_data.invoke({"spreadsheet_id": "id", "range_name": "A1"}) == "Failed to get Google Sheets service."
        assert update_sheet_data.invoke({"spreadsheet_id": "id", "range_name": "A1", "values": []}) == "Failed to get Google Sheets service."
        assert append_to_sheet.invoke({"spreadsheet_id": "id", "range_name": "A1", "values": []}) == "Failed to get Google Sheets service."

def test_sheets_api_error():
    # Mock service but raise HttpError on execute
    from googleapiclient.errors import HttpError
    from unittest.mock import MagicMock
    
    mock_resp = MagicMock()
    mock_resp.reason = 'Bad Request'
    mock_resp.status = 400
    error = HttpError(mock_resp, b'Error content')
    
    with patch("michaelmas.tools.sheets.get_sheets_service") as mock_get_service:
        mock_service = mock_get_service.return_value
        mock_service.spreadsheets.return_value.create.return_value.execute.side_effect = error
        
        assert "An error occurred" in create_spreadsheet.invoke("title")

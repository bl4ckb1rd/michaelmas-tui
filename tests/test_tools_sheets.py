from unittest.mock import MagicMock
from michaelmas.tools import sheets


def test_get_sheets_service_token_exists(mocker):
    mocker.patch("os.path.exists", return_value=True)
    mock_creds = mocker.patch("michaelmas.tools.sheets.Credentials")
    mock_build = mocker.patch("michaelmas.tools.sheets.build")

    service = sheets.get_sheets_service()

    mock_creds.from_authorized_user_file.assert_called()
    mock_build.assert_called_with(
        "sheets", "v4", credentials=mock_creds.from_authorized_user_file.return_value
    )
    assert service == mock_build.return_value


def test_create_spreadsheet(mocker):
    mock_service = MagicMock()
    mocker.patch(
        "michaelmas.tools.sheets.get_sheets_service", return_value=mock_service
    )

    mock_service.spreadsheets().create().execute.return_value = {
        "spreadsheetId": "123",
        "spreadsheetUrl": "http://sheets/123",
    }

    result = sheets.create_spreadsheet.invoke("Title")

    assert "ID 123" in result
    assert "URL http://sheets/123" in result


def test_get_sheet_data(mocker):
    mock_service = MagicMock()
    mocker.patch(
        "michaelmas.tools.sheets.get_sheets_service", return_value=mock_service
    )

    mock_service.spreadsheets().values().get().execute.return_value = {
        "values": [["A", "B"]]
    }

    result = sheets.get_sheet_data.invoke(
        {"spreadsheet_id": "123", "range_name": "A1:B1"}
    )

    assert result == [["A", "B"]]


def test_update_sheet_data(mocker):
    mock_service = MagicMock()
    mocker.patch(
        "michaelmas.tools.sheets.get_sheets_service", return_value=mock_service
    )

    mock_service.spreadsheets().values().update().execute.return_value = {
        "updatedCells": 5
    }

    result = sheets.update_sheet_data.invoke(
        {"spreadsheet_id": "123", "range_name": "A1", "values": [["X"]]}
    )

    assert "5 cells updated" in result


def test_append_to_sheet(mocker):
    mock_service = MagicMock()
    mocker.patch(
        "michaelmas.tools.sheets.get_sheets_service", return_value=mock_service
    )

    mock_service.spreadsheets().values().append().execute.return_value = {
        "updates": {"updatedRange": "Sheet1!A1"}
    }

    result = sheets.append_to_sheet.invoke(
        {"spreadsheet_id": "123", "range_name": "Sheet1", "values": [["X"]]}
    )

    assert "Appended data to Sheet1!A1" in result


def test_tool_error_handling(mocker):
    mocker.patch("michaelmas.tools.sheets.get_sheets_service", return_value=None)
    result = sheets.create_spreadsheet.invoke("Title")
    assert "Failed to get Google Sheets service" in result

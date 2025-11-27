from unittest.mock import mock_open
import json
from michaelmas.core import storage


def test_ensure_dir(mocker):
    mock_exists = mocker.patch("os.path.exists", return_value=False)
    mock_makedirs = mocker.patch("os.makedirs")

    storage._ensure_dir()

    mock_exists.assert_called_once_with(storage.CONVERSATIONS_DIR)
    mock_makedirs.assert_called_once_with(storage.CONVERSATIONS_DIR)


def test_save_conversation(mocker):
    mocker.patch("michaelmas.core.storage._ensure_dir")
    mocker.patch("os.path.exists", return_value=False)
    mock_file = mock_open()
    mocker.patch("builtins.open", mock_file)

    messages = [{"type": "human", "content": "hello"}]
    conv_id = storage.save_conversation("test-id", messages)

    assert conv_id == "test-id"
    mock_file.assert_called()
    # Verify json dump
    handle = mock_file()
    handle.write.assert_called()


def test_load_conversation(mocker):
    mocker.patch("os.path.exists", return_value=True)
    mock_data = {"id": "test-id", "messages": []}
    mock_file = mock_open(read_data=json.dumps(mock_data))
    mocker.patch("builtins.open", mock_file)

    data = storage.load_conversation("test-id")

    assert data == mock_data


def test_load_conversation_not_found(mocker):
    mocker.patch("os.path.exists", return_value=False)
    data = storage.load_conversation("non-existent")
    assert data is None


def test_list_conversations(mocker):
    mocker.patch("michaelmas.core.storage._ensure_dir")
    mocker.patch("os.listdir", return_value=["1.json", "2.json", "ignore.txt"])
    mocker.patch("os.path.exists", return_value=True)

    mock_data_1 = {"id": "1", "title": "Chat 1", "updated_at": "2023-01-01T10:00:00"}
    mock_data_2 = {"id": "2", "title": "Chat 2", "updated_at": "2023-01-02T10:00:00"}

    # Mock open to return different data for different files
    def side_effect(filename, mode):
        if "1.json" in filename:
            return mock_open(read_data=json.dumps(mock_data_1)).return_value
        if "2.json" in filename:
            return mock_open(read_data=json.dumps(mock_data_2)).return_value
        return mock_open().return_value

    mocker.patch("builtins.open", side_effect=side_effect)

    convs = storage.list_conversations()

    assert len(convs) == 2
    assert convs[0]["id"] == "2"  # Sorted by date desc
    assert convs[1]["id"] == "1"


def test_calculate_monthly_cost(mocker):
    mocker.patch("michaelmas.core.storage._ensure_dir")
    mocker.patch("os.listdir", return_value=["current.json", "old.json"])

    from datetime import datetime

    now = datetime.now()
    current_iso = now.isoformat()
    old_iso = "2020-01-01T10:00:00"

    mock_data_current = {
        "updated_at": current_iso,
        "messages": [{"cost": 0.5}, {"cost": 0.2}],
    }
    mock_data_old = {"updated_at": old_iso, "messages": [{"cost": 100.0}]}

    def side_effect(filename, mode):
        if "current.json" in filename:
            return mock_open(read_data=json.dumps(mock_data_current)).return_value
        if "old.json" in filename:
            return mock_open(read_data=json.dumps(mock_data_old)).return_value
        return mock_open().return_value

    mocker.patch("builtins.open", side_effect=side_effect)

    total = storage.calculate_monthly_cost()
    assert total == 0.7

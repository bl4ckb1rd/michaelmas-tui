import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any

CONVERSATIONS_DIR = ".conversations"

def _ensure_dir():
    if not os.path.exists(CONVERSATIONS_DIR):
        os.makedirs(CONVERSATIONS_DIR)

def save_conversation(conversation_id: str, messages: List[Dict[str, Any]], title: str = None) -> str:
    """Saves a conversation to a JSON file."""
    _ensure_dir()
    
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    file_path = os.path.join(CONVERSATIONS_DIR, f"{conversation_id}.json")
    
    # If title is not provided, try to generate one from the first message or keep existing
    if not title:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
                title = data.get("title", "New Conversation")
        else:
            # Simple title generation: first 30 chars of first human message
            title = "New Conversation"
            for msg in messages:
                if msg.get("type") == "human":
                    title = msg.get("content", "")[:30] + "..."
                    break
    
    data = {
        "id": conversation_id,
        "title": title,
        "updated_at": datetime.now().isoformat(),
        "messages": messages
    }
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
        
    return conversation_id

def load_conversation(conversation_id: str) -> Dict[str, Any]:
    """Loads a conversation from a JSON file."""
    file_path = os.path.join(CONVERSATIONS_DIR, f"{conversation_id}.json")
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, "r") as f:
        return json.load(f)

def list_conversations() -> List[Dict[str, str]]:
    """Lists all conversations, sorted by updated_at desc."""
    _ensure_dir()
    conversations = []
    for filename in os.listdir(CONVERSATIONS_DIR):
        if filename.endswith(".json"):
            file_path = os.path.join(CONVERSATIONS_DIR, filename)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    conversations.append({
                        "id": data["id"],
                        "title": data.get("title", "Untitled"),
                        "updated_at": data.get("updated_at", "")
                    })
            except Exception:
                continue # Skip broken files
                
    # Sort by updated_at descending
    conversations.sort(key=lambda x: x["updated_at"], reverse=True)
    return conversations

def delete_conversation(conversation_id: str):
    """Deletes a conversation file."""
    file_path = os.path.join(CONVERSATIONS_DIR, f"{conversation_id}.json")
    if os.path.exists(file_path):
        os.remove(file_path)

def calculate_monthly_cost() -> float:
    """Calculates the total cost of all conversations in the current month."""
    _ensure_dir()
    total_cost = 0.0
    now = datetime.now()
    current_month = now.month
    current_year = now.year

    for filename in os.listdir(CONVERSATIONS_DIR):
        if filename.endswith(".json"):
            file_path = os.path.join(CONVERSATIONS_DIR, filename)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    
                    # Check date
                    updated_at_str = data.get("updated_at")
                    if updated_at_str:
                        updated_at = datetime.fromisoformat(updated_at_str)
                        if updated_at.month == current_month and updated_at.year == current_year:
                            # Sum costs from messages
                            for msg in data.get("messages", []):
                                total_cost += msg.get("cost", 0.0)
            except Exception:
                continue
    
    return total_cost

import hashlib
import json
import os
import re
import secrets
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv(Path(__file__).parent / ".env")

app = FastAPI(title="JARVIS AI Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).parent
ADMIN_PAGE = BASE / "static" / "admin.html"
LOGIN_ART = BASE / "static" / "login-art.jpg"
STATE_PATH = BASE / "data" / "state.json"
STARTED = datetime.now()
LOG_LIMIT = 100
INBOX_LIMIT = 100
CHANNELS = {"gmail", "whatsapp", "notification"}
ROLES = {"business", "operational", "blocked"}
REPLY_MODES = {"approval", "auto", "both"}

DEFAULT_SETTINGS = {
    "name": "JARVIS",
    "model": "gpt-4o-mini",
    "reply_mode": "approval",
    "emergency_stop": False,
    "system_prompt": (
        "You are JARVIS, a calm assistant for business messages. "
        "Reply only to people on the approved list, and only after the active reply mode allows it. "
        "Never reply to operational, blocked, unknown, or spam contacts. "
        "Answer in the language the user writes in."
    ),
}

_lock = Lock()
_log: deque[dict] = deque(maxlen=LOG_LIMIT)
_contacts: list[dict] = []
_inbox: deque[dict] = deque(maxlen=INBOX_LIMIT)
_audit: deque[dict] = deque(maxlen=50)
_next_id = 1
_next_contact_id = 1
_next_inbox_id = 1
_handled = 0
_settings = dict(DEFAULT_SETTINGS)
_account: dict | None = None


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class SettingsIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    model: str = Field(min_length=1, max_length=80)
    system_prompt: str = Field(min_length=1, max_length=2000)


class ContactIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    handle: str = Field(min_length=1, max_length=120)
    channel: str
    role: str
    auto_reply: bool = False
    vip: bool = False
    reply_days: int | None = Field(default=None, ge=0, le=30)


class ControlIn(BaseModel):
    reply_mode: str
    emergency_stop: bool


class InboxIn(BaseModel):
    channel: str
    sender: str = Field(min_length=1, max_length=80)
    handle: str = Field(default="", max_length=120)
    body: str = Field(min_length=1, max_length=4000)


class AllowIn(BaseModel):
    role: str = "business"
    auto_reply: bool = False


class AccountIn(BaseModel):
    email: str = Field(min_length=3, max_length=120)
    password: str = Field(min_length=5, max_length=80)
    name: str = Field(default="", max_length=80)


def _load_state() -> None:
    global _next_id, _next_contact_id, _next_inbox_id, _handled, _settings, _account
    if not STATE_PATH.exists():
        return
    try:
        saved = json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return
    _settings = {**DEFAULT_SETTINGS, **saved.get("settings", {})}
    _next_id = int(saved.get("next_id", 1))
    _next_contact_id = int(saved.get("next_contact_id", 1))
    _next_inbox_id = int(saved.get("next_inbox_id", 1))
    _handled = int(saved.get("handled", 0))
    _contacts[:] = list(saved.get("contacts", []))
    _inbox.clear()
    for entry in reversed(saved.get("messages", [])[:LOG_LIMIT]):
        _log.appendleft(entry)
    for entry in reversed(saved.get("inbox", [])[:INBOX_LIMIT]):
        _inbox.appendleft(entry)
    _audit.clear()
    for entry in reversed(saved.get("audit", [])[:50]):
        _audit.appendleft(entry)
    saved_account = saved.get("account")
    _account = dict(saved_account) if isinstance(saved_account, dict) else None


def _save_state() -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "next_id": _next_id,
        "next_contact_id": _next_contact_id,
        "next_inbox_id": _next_inbox_id,
        "handled": _handled,
        "settings": _settings,
        "messages": list(_log),
        "contacts": list(_contacts),
        "inbox": list(_inbox),
        "audit": list(_audit),
        "account": _account,
    }
    STATE_PATH.write_text(json.dumps(payload, indent=2))


def _clean_channel(channel: str) -> str:
    value = channel.strip().lower()
    if value not in CHANNELS:
        raise HTTPException(status_code=400, detail="Channel must be gmail, whatsapp, or notification.")
    return value


def _clean_role(role: str) -> str:
    value = role.strip().lower()
    if value not in ROLES:
        raise HTTPException(status_code=400, detail="Role must be business or operational.")
    return value


def _normalize(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\u0900-\u097F\s']", "", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _local_command(text: str) -> str | None:
    asked = _normalize(text)
    hindi = bool(re.search(r"[\u0900-\u097F]", text)) or any(
        word in asked.split() for word in ("samay", "baja", "tarikh", "madad", "namaste", "haal")
    )
    now = datetime.now()

    if asked in {"hi", "hello", "hey", "namaste", "good morning", "good evening"}:
        name = _settings["name"]
        return f"Namaste. {name} online hai." if hindi else f"Hello. {name} is online."

    if asked in {"time", "what time", "what time is it", "whats the time", "samay", "kitna baja", "kitna baja hai", "time kya hai"}:
        clock = now.strftime("%I:%M %p")
        return f"Abhi samay {clock} hai." if hindi else f"The time is {clock}."

    if asked in {"date", "what is the date", "whats the date", "tarikh", "aaj ki tarikh"}:
        today = now.strftime("%A, %d %B %Y")
        return f"Aaj {today} hai." if hindi else f"Today is {today}."

    if asked in {"help", "madad", "what can you do"}:
        if hindi:
            return "Main sirf unhi ko reply karta hoon jinhe aap business list mein rakhte ho. Operational contacts ko chhod deta hoon."
        return "I reply only to people on your business list. Operational contacts are skipped."

    if asked in {"status", "uptime", "haal"}:
        elapsed = int((now - STARTED).total_seconds())
        hours, rest = divmod(elapsed, 3600)
        minutes, seconds = divmod(rest, 60)
        clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        if hindi:
            return f"Online hoon. Uptime {clock}. Messages handled: {_handled}."
        return f"Online. Uptime {clock}. Messages handled: {_handled}."

    return None


def _complete(text: str, *, with_history: bool) -> str:
    if OpenAI is None:
        raise RuntimeError("OpenAI package is not installed")
    with _lock:
        history = list(_log)[:8] if with_history else []
        settings = dict(_settings)
    history.reverse()
    messages = [{"role": "system", "content": settings["system_prompt"]}]
    for item in history:
        messages.append({"role": "user", "content": item["text"]})
        messages.append({"role": "assistant", "content": item["reply"]})
    messages.append({"role": "user", "content": text})
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=settings["model"],
        messages=messages,
        max_completion_tokens=300,
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("empty reply")
    return content


def _local_draft(sender: str, body: str) -> str:
    first = sender.strip().split()[0] if sender.strip() else "there"
    unsure = re.search(r"\b(price|cost|quote|kitna|kab|when|otp|password|account)\b", body.lower())
    if re.search(r"[\u0900-\u097F]", body):
        if unsure:
            return f"Namaste {first}, message mil gaya. Exact detail confirm karke hi jawab dunga. Main andaza nahi lagaunga."
        return f"Namaste {first}, aapka message mil gaya. Main isko dekh kar jaldi aapko update karunga."
    if unsure:
        return f"Hello {first}, I have your message. I will confirm the exact detail before answering. I will not guess."
    return f"Hello {first}, thank you for writing. I have your message and will come back with the next step shortly."


def _draft_reply(sender: str, body: str, channel: str) -> tuple[str, str]:
    prompt = (
        f"Write a short professional reply to this {channel} message from {sender}. "
        "Use the same language as the message. Do not mention operational staff.\n\n"
        f"{body}"
    )
    if os.getenv("OPENAI_API_KEY"):
        try:
            return _complete(prompt, with_history=False), "ai"
        except Exception:
            return _local_draft(sender, body), "local"
    return _local_draft(sender, body), "local"


def _reply(text: str) -> tuple[str, str]:
    command = _local_command(text)
    if command:
        return command, "local"
    if os.getenv("OPENAI_API_KEY"):
        try:
            return _complete(text, with_history=True), "ai"
        except Exception:
            return (
                "I could not reach the AI service. Ask for the time, date, status, or help.",
                "local",
            )
    return (
        "Local mode is on. I reply only to people on your business list and skip operational contacts.",
        "local",
    )


def _record_unlocked(text: str, reply: str, source: str) -> None:
    global _next_id, _handled
    _log.appendleft(
        {
            "id": _next_id,
            "text": text,
            "reply": reply,
            "source": source,
            "time": datetime.now().isoformat(),
        }
    )
    _next_id += 1
    _handled += 1


def _record(text: str, reply: str, source: str) -> None:
    with _lock:
        _record_unlocked(text, reply, source)
        _save_state()


def _find_contact(sender: str, handle: str, channel: str) -> dict | None:
    handle_key = handle.strip().lower()
    name_key = sender.strip().lower()
    with _lock:
        contacts = [dict(item) for item in _contacts]
    for contact in contacts:
        if contact["channel"] == channel and handle_key and contact["handle"].lower() == handle_key:
            return contact
    for contact in contacts:
        if contact["channel"] == channel and name_key and contact["name"].lower() == name_key:
            return contact
    return None


def _duplicate(channel: str, handle: str, body: str) -> dict | None:
    handle_key = handle.strip().lower()
    for item in _inbox:
        if (
            item["channel"] == channel
            and item["handle"].lower() == handle_key
            and item["body"] == body
        ):
            return dict(item)
    return None


def _priority(body: str, contact: dict | None) -> str:
    text = body.lower()
    if any(word in text for word in ("lottery", "you won", "crypto giveaway", "claim your prize")):
        return "spam"
    if any(word in text for word in ("otp", "password", "aadhaar", "aadhar", "cvv", "account number", "pan card")):
        return "sensitive"
    if any(word in text for word in ("urgent", "asap", "jaldi", "turant")):
        return "urgent"
    if contact and contact.get("role") == "business":
        return "business"
    return "personal"


def _in_reply_window(contact: dict | None) -> bool:
    if not contact or contact.get("role") != "business" or not contact.get("auto_reply"):
        return False
    until = contact.get("reply_until")
    if not isinstance(until, str) or not until:
        return False
    try:
        end = datetime.fromisoformat(until)
    except ValueError:
        return False
    return datetime.now() < end


def _channel_spoken(channel: str) -> str:
    if channel == "whatsapp":
        return "WhatsApp"
    if channel == "gmail":
        return "Gmail"
    return "Alert"


def _speak_for(item: dict, contact: dict | None) -> str:
    channel = _channel_spoken(item.get("channel", ""))
    sender = item.get("sender") or "Koi"
    if contact is None:
        return f"Naya {channel}. {sender} list mein nahi hai."
    if item.get("status") == "skipped":
        return f"{sender} operational hai. Reply nahi diya."
    if _phone_can_send(item, contact):
        return f"{sender} ko {channel} reply bhej raha hoon."
    if item.get("status") == "pending":
        return f"{channel} message. {sender}. Aapki approval baaki hai."
    return f"{channel} message. {sender}."


def _phone_can_send(item: dict, contact: dict | None) -> bool:
    return bool(
        contact
        and _in_reply_window(contact)
        and item.get("status") == "replied"
        and item.get("draft")
        and item.get("priority") != "sensitive"
        and not _settings.get("emergency_stop")
    )


def _queue_allowed(contact: dict) -> bool:
    if _settings.get("emergency_stop"):
        return False
    if _in_reply_window(contact):
        return True
    mode = _settings.get("reply_mode", "approval")
    if mode == "auto":
        return True
    if mode == "both":
        return bool(contact.get("auto_reply") or contact.get("vip"))
    return False


def _decide(item: dict, contact: dict | None, draft: str, source: str) -> None:
    if item.get("priority") == "spam":
        item.update(
            decision="skip",
            status="skipped",
            draft="",
            draft_source="local",
            reason="Spam. No reply.",
        )
        return
    if contact is None:
        item.update(
            decision="hold",
            status="held",
            draft="",
            draft_source="local",
            reason="Naya contact. List mein nahi, isliye reply nahi diya.",
        )
        return
    if contact["role"] in {"operational", "blocked"}:
        item.update(
            decision="skip",
            status="skipped",
            draft="",
            draft_source="local",
            reason="Operational contact. No reply." if contact["role"] == "operational" else "Blocked contact. No reply.",
        )
        return
    if _settings.get("emergency_stop"):
        item.update(
            decision="hold",
            status="held",
            draft=draft,
            draft_source=source,
            reason="Emergency stop. No reply will be sent.",
        )
        return
    if item.get("priority") == "sensitive":
        item.update(
            decision="reply",
            status="pending",
            draft=draft,
            draft_source=source,
            reason="Sensitive message. Extra approval required. No guess was sent.",
        )
        return
    queued = _queue_allowed(contact)
    item.update(
        decision="reply",
        status="replied" if queued else "pending",
        draft=draft,
        draft_source=source,
        reason=(
            "Aapke chune hue dinon tak reply queue hua."
            if queued and _in_reply_window(contact)
            else "Approved contact. Reply queued."
            if queued
            else "Approved contact. Waiting for your approval."
        ),
    )


def _mode() -> str:
    return "openai" if os.getenv("OPENAI_API_KEY") else "local"


def _other_reply(sender: str, body: str) -> str:
    first = sender.strip().split()[0] if sender.strip() else "there"
    hindi = bool(re.search(r"[\u0900-\u097F]", body))
    if hindi:
        return f"English meaning of the reply: Hello {first}, I have your message. I will confirm the detail before answering."
    return f"Jawab ka Hindi matlab: Namaste {first}, message mil gaya. Detail confirm karke hi jawab dunga."


def _with_explain(item: dict) -> dict:
    copy = dict(item)
    body = copy.get("body") or ""
    hindi = bool(re.search(r"[\u0900-\u097F]", body))
    copy["lang"] = "hindi" if hindi else "english"
    if copy.get("draft"):
        copy["explain"] = _other_reply(copy.get("sender") or "", body)
    elif hindi:
        copy["explain"] = "Hindi message. English meaning is not guessed."
    else:
        copy["explain"] = "English message. Hindi meaning appears when a draft exists."
    return copy


def _overview() -> dict:
    with _lock:
        return {
            "assistant": _settings["name"],
            "status": "online",
            "mode": _mode(),
            "started_at": STARTED.isoformat(),
            "uptime_seconds": int((datetime.now() - STARTED).total_seconds()),
            "handled": _handled,
            "settings": dict(_settings),
            "messages": list(_log),
            "contacts": list(_contacts),
            "inbox": [_with_explain(item) for item in _inbox],
            "audit": list(_audit),
            "synced_at": datetime.now().isoformat(),
        }


_load_state()


@app.get("/")
def home():
    with _lock:
        name = _settings["name"]
    return {
        "assistant": name,
        "status": "online",
        "mode": _mode(),
        "message": f"Hello. {name} is ready.",
        "time": datetime.now().isoformat(),
    }


def _set_control(reply_mode: str | None = None, emergency_stop: bool | None = None) -> None:
    with _lock:
        if reply_mode is not None:
            if reply_mode not in REPLY_MODES:
                raise HTTPException(status_code=400, detail="Reply mode must be approval, auto, or both.")
            _settings["reply_mode"] = reply_mode
        if emergency_stop is not None:
            _settings["emergency_stop"] = emergency_stop
        _save_state()


def _summary() -> str:
    with _lock:
        waiting = [
            item for item in _inbox
            if item.get("status") == "pending" and item.get("priority", "business") in {"business", "urgent"}
        ]
    if not waiting:
        return "Koi naya business message approval ka wait nahi kar raha."
    lines = [f"{item['channel']}: {item['sender']} — {item['body'][:90]}" for item in waiting[:6]]
    return "Approval mein business messages:\n" + "\n".join(lines)


def _apply_order(text: str) -> str | None:
    asked = _normalize(text)
    if any(phrase in asked for phrase in ("emergency stop", "stop all replies", "sab band", "sab reply band")):
        _set_control(emergency_stop=True)
        return "Emergency stop on. Kisi ko reply nahi jayega."
    if any(phrase in asked for phrase in ("resume replies", "emergency off", "replies chalu", "stop hatao")):
        _set_control(emergency_stop=False)
        return "Emergency stop off. Approved replies phir se approval mode par hain."
    if any(phrase in asked for phrase in ("dono mode", "both modes", "manual aur automatic")):
        _set_control(reply_mode="both")
        return "Dono modes on. VIP ya auto contacts queue honge. Baaki approval ka wait karenge."
    if any(phrase in asked for phrase in ("automatic replies band", "auto replies band", "automatic reply band", "auto band", "automatic replies off", "auto replies off")):
        _set_control(reply_mode="approval")
        return "Automatic replies band. Har reply pehle aapki approval ka wait karega."
    if any(phrase in asked for phrase in ("sirf approved", "automatic reply", "auto reply")):
        _set_control(reply_mode="auto")
        return "Sirf approved business contacts ko automatic reply hoga. Baaki ko kuch nahi jayega."
    if any(phrase in asked for phrase in ("pehle meri approval", "meri approval", "approval mode")):
        _set_control(reply_mode="approval")
        return "Ab har reply pehle aapki approval ka wait karega."
    if any(phrase in asked for phrase in ("pending email", "pending emails", "pending gmail")):
        return _pending_emails()
    if any(phrase in asked for phrase in ("approved contacts", "approved contact", "reply list", "contact list")):
        return _contact_list()
    named = _named_reply(asked)
    if named:
        return named
    if any(phrase in asked for phrase in ("summary", "naye business", "business emails", "business messages", "daily briefing", "briefing")):
        return _briefing()
    return None


def _briefing() -> str:
    with _lock:
        items = list(_inbox)
    business = [item for item in items if item.get("priority") in {"business", "urgent", "sensitive"}]
    pending = [item for item in items if item.get("status") == "pending"]
    spam = [item for item in items if item.get("priority") == "spam"]
    lines = [
        f"Aaj desk par {len(items)} messages hain.",
        f"Business {len(business)}. Pending approval {len(pending)}. Spam {len(spam)}.",
    ]
    lines.extend(f"- {item['sender']}: {item['body'][:80]}" for item in pending[:5])
    if not pending:
        lines.append("Koi pending reply nahi.")
    return "\n".join(lines)


def _pending_emails() -> str:
    with _lock:
        items = [item for item in _inbox if item.get("channel") == "gmail" and item.get("status") == "pending"]
    if not items:
        return "Koi pending email nahi."
    lines = [f"{item['sender']}: {item['body'][:80]}" for item in items[:6]]
    return "Pending emails:\n" + "\n".join(lines)


def _contact_list() -> str:
    with _lock:
        names = [contact["name"] for contact in _contacts if contact.get("role") == "business"]
    if not names:
        return "Approved contact list khali hai."
    return "Approved contacts: " + ", ".join(names)


def _named_reply(asked: str) -> str | None:
    if " ko " not in asked or ("reply" not in asked and "jawab" not in asked):
        return None
    name = asked.split(" ko ", 1)[0].replace("jarvis", "").replace("please", "").strip()
    if not name or len(name) > 40:
        return None
    with _lock:
        matches = [item for item in _inbox if name in item.get("sender", "").lower()]
    if not matches:
        return f"{name.title()} ka koi message desk par nahi hai. Main andaza laga kar jawab nahi likhunga."
    item = matches[0]
    if not item.get("draft"):
        return f"{item['sender']} approved list par nahi. Unhe reply nahi likha jayega."
    return f"{item['sender']} ke liye draft, abhi bheja nahi:\n{item['draft']}"


def _remember(item: dict) -> None:
    _audit.appendleft(
        {
            "sender": item.get("sender", ""),
            "channel": item.get("channel", ""),
            "body": (item.get("body") or "")[:160],
            "draft": (item.get("draft") or "")[:200],
            "time": datetime.now().isoformat(),
        }
    )


@app.post("/message")
def receive_message(data: Message):
    text = data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message is empty.")
    order = _apply_order(text)
    if order:
        _record(text, order, "local")
        return {"received": text, "reply": order, "source": "local"}
    reply, source = _reply(text)
    _record(text, reply, source)
    return {"received": text, "reply": reply, "source": source}


@app.post("/api/control")
def update_control(body: ControlIn):
    _set_control(reply_mode=body.reply_mode, emergency_stop=body.emergency_stop)
    with _lock:
        return {
            "reply_mode": _settings["reply_mode"],
            "emergency_stop": _settings["emergency_stop"],
        }


@app.get("/admin")
def admin_panel():
    return FileResponse(ADMIN_PAGE)


@app.get("/login-art.jpg")
def login_art():
    return FileResponse(LOGIN_ART)


@app.get("/admin.")
def admin_panel_with_dot():
    return RedirectResponse("/admin")


@app.get("/api/admin/overview")
def admin_overview():
    return _overview()


@app.put("/api/admin/settings")
def update_settings(body: SettingsIn):
    name = body.name.strip()
    model = body.model.strip()
    prompt = body.system_prompt.strip()
    if not name or not model or not prompt:
        raise HTTPException(status_code=400, detail="Name, model, and system prompt are required.")
    with _lock:
        _settings.update({"name": name, "model": model, "system_prompt": prompt})
        _save_state()
        saved = dict(_settings)
    return saved


@app.delete("/api/admin/log")
def clear_log():
    with _lock:
        _log.clear()
        _save_state()
    return {"cleared": True}


@app.delete("/api/admin/audit")
def clear_audit():
    with _lock:
        _audit.clear()
        _save_state()
    return {"cleared": True}


@app.post("/api/contacts")
def add_contact(body: ContactIn):
    global _next_contact_id
    channel = _clean_channel(body.channel)
    role = _clean_role(body.role)
    name = body.name.strip()
    handle = body.handle.strip()
    auto_reply = bool(body.auto_reply) and role == "business"
    vip = bool(body.vip) and role == "business"
    reply_until = None
    if body.reply_days is not None:
        if body.reply_days > 0 and role == "business":
            auto_reply = True
            reply_until = (datetime.now() + timedelta(days=body.reply_days)).isoformat(timespec="seconds")
        else:
            auto_reply = False
            reply_until = None
    with _lock:
        for contact in _contacts:
            if contact["channel"] == channel and contact["handle"].lower() == handle.lower():
                contact["name"] = name
                contact["role"] = role
                contact["auto_reply"] = auto_reply
                contact["vip"] = vip
                if body.reply_days is not None:
                    contact["reply_until"] = reply_until
                _save_state()
                return dict(contact)
        contact = {
            "id": _next_contact_id,
            "name": name,
            "handle": handle,
            "channel": channel,
            "role": role,
            "auto_reply": auto_reply,
            "vip": vip,
            "reply_until": reply_until,
        }
        _next_contact_id += 1
        _contacts.append(contact)
        _save_state()
        return dict(contact)


@app.delete("/api/contacts/{contact_id}")
def delete_contact(contact_id: int):
    with _lock:
        kept = [contact for contact in _contacts if contact["id"] != contact_id]
        if len(kept) == len(_contacts):
            raise HTTPException(status_code=404, detail="Contact not found.")
        _contacts[:] = kept
        _save_state()
    return {"deleted": True}


@app.post("/api/inbox")
def ingest_message(body: InboxIn):
    global _next_inbox_id
    channel = _clean_channel(body.channel)
    sender = body.sender.strip()
    handle = body.handle.strip() or sender
    text = body.body.strip()
    contact = _find_contact(sender, handle, channel)
    draft, source = ("", "local")
    if contact and contact["role"] == "business":
        draft, source = _draft_reply(sender, text, channel)
    with _lock:
        existing = _duplicate(channel, handle, text)
        if existing:
            return existing
        item = {
            "id": _next_inbox_id,
            "channel": channel,
            "sender": sender,
            "handle": handle,
            "body": text,
            "priority": _priority(text, contact),
            "time": datetime.now().isoformat(),
        }
        _next_inbox_id += 1
        _decide(item, contact, draft, source)
        if item["status"] == "replied":
            _record_unlocked(
                f"[{channel}] {sender}: {text}",
                item["draft"],
                item["draft_source"],
            )
            _remember(item)
        _inbox.appendleft(item)
        _save_state()
        result = dict(item)
        result["speak"] = _speak_for(item, contact)
        result["send_reply"] = item.get("draft") or "" if _phone_can_send(item, contact) else ""
        return result


def _inbox_item(item_id: int) -> dict | None:
    for item in _inbox:
        if item["id"] == item_id:
            return item
    return None


@app.post("/api/inbox/{item_id}/send")
async def send_inbox_reply(item_id: int, request: Request):
    raw = (await request.body()).decode()
    edited = ""
    if raw.strip():
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            edited = str(payload.get("draft") or "").strip()
    if len(edited) > 2000:
        raise HTTPException(status_code=400, detail="Reply is too long.")
    with _lock:
        item = _inbox_item(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Message not found.")
        if _settings.get("emergency_stop"):
            raise HTTPException(status_code=409, detail="Emergency stop is on. No reply will be sent.")
        if item["decision"] != "reply":
            raise HTTPException(
                status_code=409,
                detail="JARVIS will not message this person. They are not on your reply list.",
            )
        if edited:
            item["draft"] = edited
            item["draft_source"] = "local"
        if not item.get("draft"):
            raise HTTPException(status_code=409, detail="Write a reply before approving.")
        if item["status"] != "replied":
            item["status"] = "replied"
            item["reason"] = "Approved by you. Reply queued."
            _record_unlocked(
                f"[{item['channel']}] {item['sender']}: {item['body']}",
                item["draft"],
                item.get("draft_source") or "local",
            )
            _remember(item)
        _save_state()
        return dict(item)


@app.post("/api/inbox/{item_id}/skip")
def skip_inbox(item_id: int):
    with _lock:
        item = _inbox_item(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Message not found.")
        if item["status"] == "replied":
            raise HTTPException(status_code=409, detail="This reply is already queued.")
        item["decision"] = "skip"
        item["status"] = "skipped"
        item["draft"] = ""
        item["reason"] = "Held. JARVIS will not reply."
        _save_state()
        return dict(item)


@app.post("/api/inbox/{item_id}/allow")
def allow_sender(item_id: int, body: AllowIn):
    role = _clean_role(body.role)
    with _lock:
        item = _inbox_item(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Message not found.")
        snapshot = dict(item)
    contact = add_contact(
        ContactIn(
            name=snapshot["sender"],
            handle=snapshot["handle"],
            channel=snapshot["channel"],
            role=role,
            auto_reply=body.auto_reply,
        )
    )
    draft, source = ("", "local")
    if contact["role"] == "business":
        draft, source = _draft_reply(snapshot["sender"], snapshot["body"], snapshot["channel"])
    with _lock:
        item = _inbox_item(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Message not found.")
        _decide(item, contact, draft, source)
        if item["status"] == "replied":
            _record_unlocked(
                f"[{item['channel']}] {item['sender']}: {item['body']}",
                item["draft"],
                item["draft_source"],
            )
            _remember(item)
        _save_state()
        return dict(item)


def _page(title: str, body: str) -> HTMLResponse:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin: 0; background: #06101c; color: #e7f7ff; font: 16px/1.5 -apple-system, BlinkMacSystemFont, sans-serif; }}
    main {{ max-width: 720px; margin: 0 auto; padding: 32px 20px 64px; }}
    a {{ color: #3ce7ff; }}
    h1 {{ font-size: 28px; margin-bottom: 8px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    p, li {{ color: #d5e8f5; }}
    .muted {{ color: #7e9bb3; }}
    form {{ margin-top: 20px; display: grid; gap: 12px; max-width: 360px; }}
    input, button {{ font: inherit; border-radius: 10px; padding: 12px 14px; }}
    input {{ border: 1px solid #1a3a56; background: #0c1a2e; color: #e7f7ff; }}
    button {{ border: 0; background: #ff6b8a; color: #1a0610; font-weight: 700; }}
  </style>
</head>
<body><main>{body}</main></body>
</html>"""
    return HTMLResponse(html)


def _wipe_account() -> None:
    global _next_id, _next_contact_id, _next_inbox_id, _handled, _settings, _account
    _log.clear()
    _contacts.clear()
    _inbox.clear()
    _audit.clear()
    _next_id = 1
    _next_contact_id = 1
    _next_inbox_id = 1
    _handled = 0
    _settings = dict(DEFAULT_SETTINGS)
    _account = None
    _save_state()


def _clean_email(email: str) -> str:
    value = email.strip().lower()
    domain = value.split("@")[-1] if "@" in value else ""
    if "@" not in value or "." not in domain or " " in value:
        raise HTTPException(status_code=400, detail="Email sahi nahi hai.")
    return value


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000).hex()


def _public_account() -> dict:
    if _account is None:
        raise HTTPException(status_code=401, detail="Sign in required.")
    return {"email": _account["email"], "name": _account["name"]}


@app.post("/api/account/register")
def register_account(body: AccountIn):
    global _account
    email = _clean_email(body.email)
    name = body.name.strip() or email.split("@")[0]
    with _lock:
        if _account is not None:
            raise HTTPException(status_code=409, detail="Account pehle se hai. Sign in karo.")
        salt = secrets.token_hex(16)
        _account = {
            "email": email,
            "name": name,
            "salt": salt,
            "password_hash": _hash_password(body.password, salt),
            "token": secrets.token_urlsafe(32),
        }
        _save_state()
        return {"token": _account["token"], "email": email, "name": name}


@app.post("/api/account/login")
def login_account(body: AccountIn):
    email = _clean_email(body.email)
    with _lock:
        if _account is None or _account["email"] != email:
            raise HTTPException(status_code=401, detail="Email ya password galat hai.")
        digest = _hash_password(body.password, _account["salt"])
        if not secrets.compare_digest(digest, _account["password_hash"]):
            raise HTTPException(status_code=401, detail="Email ya password galat hai.")
        _account["token"] = secrets.token_urlsafe(32)
        _save_state()
        return {"token": _account["token"], "email": email, "name": _account["name"]}


@app.get("/api/account/me")
def account_me(request: Request):
    header = request.headers.get("authorization", "")
    token = header.removeprefix("Bearer ").strip()
    with _lock:
        if (
            _account is None
            or not token
            or len(token) != len(_account["token"])
            or not secrets.compare_digest(token, _account["token"])
        ):
            raise HTTPException(status_code=401, detail="Sign in required.")
        return _public_account()


@app.get("/privacy", response_class=HTMLResponse)
def privacy_policy():
    return _page(
        "JARVIS Privacy Policy",
        """
        <p class="muted">Last updated 17 November 2026</p>
        <h1>Privacy Policy</h1>
        <p>JARVIS is a personal communication assistant that runs on your phone and on the computer where this backend is running. The login stays on this computer. It is not a cloud account, and JARVIS does not sell your data.</p>
        <h2>Data stored on this computer</h2>
        <ul>
          <li>Contact names, handles, channel, and whether they are business, operational, or blocked.</li>
          <li>Message text you log, notification text sent from your Android phone, drafts, and the reply decision.</li>
          <li>The reply log, assistant name, reply mode, and emergency-stop setting.</li>
          <li>Your login email, display name, and a password hash. The password itself is not stored.</li>
        </ul>
        <h2>Phone permissions</h2>
        <p>On Android, notification access can read notification text from apps you allow, such as Gmail or WhatsApp, and send that text to this backend. The microphone is used only when you tap the voice button. iPhone cannot read notifications from other apps.</p>
        <h2>What is not connected</h2>
        <p>Gmail login and the WhatsApp Business API are not connected. JARVIS does not fetch your mailbox or send a WhatsApp message by itself. A draft stays on this computer until you approve it.</p>
        <h2>Optional model</h2>
        <p>If you put an OpenAI key in the backend .env file, the message text can be sent to OpenAI to write a draft. Without that key, drafts are written on this computer and are not sent to a model provider.</p>
        <h2>Delete your data</h2>
        <p>Open the <a href="/delete-account">delete account</a> page and confirm. That removes contacts, inbox messages, drafts, and the reply log from this backend. It does not delete mail or chats stored by Google or WhatsApp, and it does not uninstall the app.</p>
        """,
    )


@app.get("/delete-account", response_class=HTMLResponse)
def delete_account_page():
    return _page(
        "Delete JARVIS data",
        """
        <h1>Delete account</h1>
        <p>This deletes the JARVIS login on this computer, plus contacts, messages, drafts, and the reply log.</p>
        <p>Gmail and WhatsApp keep their own copies. The app stays installed, and you can create a new login.</p>
        <form method="post" action="/api/account/delete">
          <label for="confirm">Type DELETE to confirm</label>
          <input id="confirm" name="confirm" autocomplete="off" required>
          <button type="submit">Delete data on this computer</button>
        </form>
        <p class="muted"><a href="/privacy">Privacy Policy</a></p>
        """,
    )


@app.post("/api/account/delete", response_class=HTMLResponse)
async def delete_account(request: Request):
    raw = (await request.body()).decode()
    confirm = parse_qs(raw).get("confirm", [""])[0]
    if confirm.strip().upper() != "DELETE":
        raise HTTPException(status_code=400, detail="Type DELETE to confirm.")
    with _lock:
        _wipe_account()
    return _page(
        "JARVIS data deleted",
        """
        <h1>Data deleted</h1>
        <p>The login, contacts, inbox messages, drafts, and the reply log on this computer have been removed. Settings are back to the default approval mode.</p>
        <p><a href="/privacy">Privacy Policy</a></p>
        """,
    )

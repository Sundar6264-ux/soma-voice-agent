"""Settings from the environment, with a small .env loader (no extra dependency)."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    restaurant_file: Path
    db_path: Path
    model: str
    effort: str
    public_base_url: str
    twilio_auth_token: str
    allow_unsigned_twilio: bool
    admin_key: str
    relay_secret: str
    max_sessions: int
    relay_attrs: dict

    @property
    def ws_base_url(self) -> str:
        return "wss://" + self.public_base_url.split("://", 1)[-1]

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        env = os.environ.get
        return cls(
            restaurant_file=Path(env("RESTAURANT_FILE", str(ROOT / "restaurants" / "demo.yaml"))),
            db_path=Path(env("DB_PATH", str(ROOT / "voiceagent.db"))),
            model=env("MODEL", "claude-opus-5-5"),
            effort=env("EFFORT", "low"),
            public_base_url=env("PUBLIC_BASE_URL", "").rstrip("/"),
            twilio_auth_token=env("TWILIO_AUTH_TOKEN", ""),
            allow_unsigned_twilio=_flag("ALLOW_UNSIGNED_TWILIO"),
            admin_key=env("ADMIN_KEY", ""),
            # Only needs to outlive one call; a restart drops live calls anyway.
            relay_secret=env("RELAY_SECRET", "") or secrets.token_hex(32),
            max_sessions=int(env("MAX_SESSIONS", "4")),
            relay_attrs={
                "language": env("RELAY_LANGUAGE", "en-US"),
                "ttsProvider": env("RELAY_TTS_PROVIDER", ""),
                "voice": env("RELAY_VOICE", ""),
                "transcriptionProvider": env("RELAY_TRANSCRIPTION_PROVIDER", ""),
                "speechModel": env("RELAY_SPEECH_MODEL", ""),
            },
        )

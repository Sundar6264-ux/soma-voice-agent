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
    # Self-hosted call path (Asterisk + AssemblyAI + Inworld). Off unless ASTERISK_TOKEN is set.
    asterisk_token: str = ""
    audiosocket_port: int = 9092
    assemblyai_api_key: str = ""
    inworld_api_key: str = ""
    inworld_voice: str = "Ashley"
    inworld_model: str = "inworld-tts-1.5-mini"
    inworld_speaking_rate: float = 1.0
    inworld_max_concurrent: int = 2   # Inworld's free tier allows only a few simultaneous requests
    # AssemblyAI turn detection (README, "Latency"). Silence in ms before a turn may end / must end.
    stt_min_turn_silence_ms: int = 400
    stt_max_turn_silence_ms: int = 1280
    stt_end_of_turn_confidence: float | None = None   # AssemblyAI default 0.4 when unset
    stt_wait_for_formatting: bool = False              # True: the old behaviour, wait for the formatted turn
    # Testing only: "18:00" makes the agent's clock read that time today (restaurant timezone),
    # so orders can be tested outside opening hours. Empty in real use.
    demo_time: str = ""
    # "Soma's AI" browser persona (voiceagent/profile.py, README "Soma's AI").
    profile_resume_path: str = "/home/sundar/career-central/resumes/master-resume.md"
    profile_inworld_voice: str = ""          # empty: the same voice as the phone path
    profile_speaking_rate: float = 1.0       # the cloned voice speaks about 130 words a minute at 1.0
    # Public "Talk to my AI" on somasundar.dev (Phase 2): off unless TALK_PUBLIC=on.
    talk_public: bool = False
    transcript_email: bool = True   # email Soma the transcript after every Soma's AI conversation
    talk_per_ip_day: int = 3
    talk_per_day: int = 20
    talk_allowed_origins: tuple = ("https://somasundar.dev",)
    profile_time_limit_s: int = 600
    # Messages taken by take_message: "on" emails them, "test" emails with a [TEST] subject, "off" only logs.
    message_email: str = "off"

    @property
    def selfhost_enabled(self) -> bool:
        return bool(self.asterisk_token)

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
            asterisk_token=env("ASTERISK_TOKEN", ""),
            audiosocket_port=int(env("AUDIOSOCKET_PORT", "9092")),
            assemblyai_api_key=env("ASSEMBLYAI_API_KEY", ""),
            inworld_api_key=env("INWORLD_API_KEY", ""),
            inworld_voice=env("INWORLD_VOICE", "Ashley"),
            inworld_model=env("INWORLD_MODEL", "inworld-tts-1.5-mini"),
            inworld_speaking_rate=float(env("INWORLD_SPEAKING_RATE", "1.0")),
            inworld_max_concurrent=int(env("INWORLD_MAX_CONCURRENT", "2")),
            stt_min_turn_silence_ms=int(env("STT_MIN_TURN_SILENCE_MS", "400")),
            stt_max_turn_silence_ms=int(env("STT_MAX_TURN_SILENCE_MS", "1280")),
            stt_end_of_turn_confidence=float(env("STT_END_OF_TURN_CONFIDENCE")) if env("STT_END_OF_TURN_CONFIDENCE") else None,
            stt_wait_for_formatting=_flag("STT_WAIT_FOR_FORMATTING"),
            demo_time=env("DEMO_TIME", "").strip(),
            profile_resume_path=env("PROFILE_RESUME_PATH", "/home/sundar/career-central/resumes/master-resume.md"),
            profile_inworld_voice=env("PROFILE_INWORLD_VOICE", ""),
            profile_speaking_rate=float(env("PROFILE_SPEAKING_RATE", "1.0")),
            talk_public=_flag("TALK_PUBLIC"),
            transcript_email=env("TRANSCRIPT_EMAIL", "on").strip().lower() not in {"off", "0", "no", "false"},
            talk_per_ip_day=int(env("TALK_PER_IP_DAY", "3")),
            talk_per_day=int(env("TALK_PER_DAY", "20")),
            talk_allowed_origins=tuple(o.strip() for o in env("TALK_ALLOWED_ORIGINS", "https://somasundar.dev").split(",") if o.strip()),
            profile_time_limit_s=int(env("PROFILE_TIME_LIMIT_S", "600")),
            message_email=env("MESSAGE_EMAIL", "off").strip().lower() or "off",
        )

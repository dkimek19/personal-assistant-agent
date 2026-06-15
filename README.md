# Personal Assistant Agent

로컬 LLM(Ollama) 기반의 1인용 개인 비서 에이전트입니다. Web UI / Telegram / Discord
어디서 대화를 시작하든 동일한 세션(기억)을 공유하며, 캘린더·할일·날씨·웹검색·문서·코드실행
등의 도구를 직접 호출해 답변합니다.

## 주요 기능

- **멀티 인터페이스 통합 세션** — Web UI, Telegram, Discord에서 대화한 내용을
  하나의 `working_memory`로 공유 (`assistant/session_resolver.py`,
  `assistant/session_store.py`)
- **도구 호출 (Tool Calling)** — Ollama `/api/chat`의 tool calling을 이용해
  필요할 때 자동으로 아래 도구들을 사용 (`assistant/agent_core.py`,
  `assistant/tools/`)
  - 날씨 조회 (Open-Meteo)
  - Google Calendar 일정 조회/생성/수정/삭제
  - Google Tasks 할일 조회/생성/수정/완료
  - 웹 검색 + 요약 (SearXNG)
  - PDF/DOCX 문서 읽기 및 생성
  - Docker 샌드박스에서 코드 실행
- **슬래시 커맨드**
  - `/note` — SQLite에 메모 저장 (`assistant/notes.py`)
  - `/remember` — 장기 기억에 저장 (`assistant/long_term_memory.py`)
  - `/compress` — 대화 컨텍스트 수동 압축 (컨텍스트가 길어지면 자동 압축도 동작)
    (`assistant/compression.py`)
- **자동화 (macOS launchd)** — `assistant/launchd.py`
  - 메인 Web UI 상시 실행 (크래시 시 자동 재시작)
  - 매일 새벽 3시 SQLite 백업 + 30일 지난 백업 자동 삭제
  - 5분마다 캘린더 일정 30분 전 Telegram 알림
  - 1시간마다 디스크 사용량 확인, 20GB 초과 시 Telegram 경고

## 요구 사항

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (패키지/가상환경 관리)
- [Ollama](https://ollama.com/)가 로컬에서 실행 중이고, 사용할 모델이 받아져 있어야 함
  (기본값: `gemma4:12b-mlx`)
- Docker (코드 실행 도구 사용 시)
- SearXNG 인스턴스 (웹 검색 도구 사용 시)
- Google Cloud OAuth 클라이언트 (Calendar/Tasks 도구 사용 시)

## 설치

```bash
uv sync
```

## 환경 변수 (`.env`)

프로젝트 루트에 `.env` 파일을 만들고 아래 값을 채워주세요 (`.gitignore`에 의해
커밋되지 않습니다).

| 변수 | 설명 | 기본값 |
|---|---|---|
| `OLLAMA_URL` | Ollama 서버 주소 | `http://localhost:11434` |
| `OLLAMA_MODEL` | 사용할 모델명 | `gemma4:12b-mlx` |
| `SEARXNG_URL` | SearXNG 인스턴스 주소 | `http://localhost:8888` |
| `TELEGRAM_TOKEN` | Telegram 봇 토큰 (Telegram 인터페이스 사용 시) | - |
| `DISCORD_TOKEN` | Discord 봇 토큰 (Discord 인터페이스 사용 시) | - |
| `ASSISTANT_CREDENTIALS_DIR` | Google OAuth `credentials.json`/토큰 파일 위치 | `~/assistant/credentials` |

Google Calendar/Tasks를 사용하려면 `ASSISTANT_CREDENTIALS_DIR` 아래에
Google Cloud Console에서 받은 `credentials.json`을 넣어두세요. 최초 호출 시
OAuth 인증 흐름을 거쳐 토큰 파일이 같은 디렉터리에 생성됩니다.

## 실행

### Web UI

```bash
uv run python -m assistant.main
```

`http://127.0.0.1:8000` 에서 채팅 UI와 날씨/캘린더/메모 위젯을 확인할 수 있습니다.

### Telegram 봇

```bash
uv run python -m assistant.interfaces.telegram_bot
```

### Discord 봇

```bash
uv run python -m assistant.interfaces.discord_bot
```

## 백그라운드 자동화 등록 (macOS)

```bash
uv run python -c "from assistant.launchd import install_all; install_all()"
```

메인 에이전트(`com.personalassistant.agent`), 백업(`com.personalassistant.backup`),
캘린더 알림(`com.personalassistant.calendar-alerts`),
디스크 모니터(`com.personalassistant.disk-monitor`) 4개의 launchd job이
`~/Library/LaunchAgents`에 등록되고 즉시 로드됩니다.

## 테스트

```bash
uv run pytest -q
```

대부분의 테스트는 외부 서비스 없이 동작합니다. Ollama/SearXNG 라이브 연동이 필요한
SLA·통합 테스트 일부는 해당 서비스가 없으면 자동으로 skip됩니다.

## 프로젝트 구조

```
assistant/
  agent_core.py       # 모든 인터페이스의 공유 진입점 (handle_user_message)
  session_store.py    # SQLite 기반 공유 working_memory / 장기 기억
  session_resolver.py # 인터페이스별 사용자 -> 공통 user_id/session_id 매핑
  notes.py            # /note
  long_term_memory.py # /remember
  compression.py      # /compress, 자동 컨텍스트 압축
  backup.py           # DB 백업/정리
  calendar_alerts.py  # 캘린더 알림 (Telegram)
  disk_monitor.py      # 디스크 사용량 경고 (Telegram)
  launchd.py          # macOS launchd plist 생성/설치
  interfaces/         # web_ui (FastAPI), telegram_bot, discord_bot
  tools/              # weather, calendar, tasks, searxng, documents, code_execution
tests/                # pytest 테스트 (836 passed, 3 skipped)
```

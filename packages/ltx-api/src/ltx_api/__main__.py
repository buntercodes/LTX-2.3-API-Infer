import uvicorn

from ltx_api.config import settings


def main() -> None:
    uvicorn.run(
        "ltx_api.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
